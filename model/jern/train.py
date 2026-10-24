import math
import os
import pickle
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from constants import CHAR_OUT_DIR, DATA_DIR, DATASET, META_FILE, NAME, SEED
from model.jern.config import RuntimeConfig, seed_everything
from model.jern.network import JERN, JERNConfig, Checkpoint


@dataclass
class TrainConfig(RuntimeConfig):
    out_dir: str = CHAR_OUT_DIR
    data_dir: str = DATA_DIR
    eval_interval: int = 250
    log_interval: int = 50
    eval_iters: int = 20
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = 'scratch'
    wandb_log: bool = False
    wandb_project: str = NAME.lower()
    wandb_run_name: str = DATASET
    dataset: str = DATASET
    gradient_accumulation_steps: int = 1
    batch_size: int = 16
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    learning_rate: float = 1e-3
    max_iters: int = 2000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 100
    lr_decay_iters: int = 2000
    min_lr: float = 1e-4
    backend: str = 'nccl'


class DistributedContext:

    def __init__(self, backend):
        self.enabled = int(os.environ.get('RANK', -1)) != -1
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        if self.enabled:
            init_process_group(backend=backend)
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])

    @property
    def is_master(self):
        return self.rank == 0

    @property
    def device(self):
        return f'cuda:{self.local_rank}'

    def wrap(self, model):
        return DDP(model, device_ids=[self.local_rank]) if self.enabled else model

    def close(self):
        if self.enabled:
            destroy_process_group()


class BatchLoader:

    def __init__(self, data_dir, block_size, batch_size, device):
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.pin_memory = 'cuda' in device

    def get_batch(self, split):
        data = np.memmap(os.path.join(self.data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if self.pin_memory:
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)

    def vocab_size(self):
        with open(os.path.join(self.data_dir, META_FILE), 'rb') as f:
            return pickle.load(f)['vocab_size']


class LearningRateScheduler:

    def __init__(self, learning_rate, min_lr, warmup_iters, decay_iters, decay):
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
        self.decay = decay

    def get_lr(self, it):
        if not self.decay:
            return self.learning_rate
        if it < self.warmup_iters:
            return self.learning_rate * (it + 1) / (self.warmup_iters + 1)
        if it > self.decay_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / (self.decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)


class Trainer:

    def __init__(self, config):
        self.config = config
        self.config_snapshot = config.to_dict()
        self.ddp = DistributedContext(config.backend)
        if self.ddp.enabled:
            config.device = self.ddp.device
            torch.cuda.set_device(config.device)
            assert config.gradient_accumulation_steps % self.ddp.world_size == 0
            config.gradient_accumulation_steps //= self.ddp.world_size
        tokens_per_iter = config.gradient_accumulation_steps * self.ddp.world_size * config.batch_size * config.block_size
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        if self.ddp.is_master:
            os.makedirs(config.out_dir, exist_ok=True)
        seed_everything(SEED + self.ddp.rank)
        self.ctx = config.autocast()
        self.loader = BatchLoader(os.path.join(config.data_dir, config.dataset), config.block_size, config.batch_size, config.device)
        self.iter_num = 0
        self.local_iter_num = 0
        self.best_val_loss = 1e9
        self.running_mfu = -1.0
        self.checkpoint = None
        self.raw_model = self.build_model()
        self.model_args = asdict(self.raw_model.config)
        self.scaler = torch.amp.GradScaler(
            config.device_type, enabled=(config.dtype == 'float16' and config.device_type == 'cuda'))
        self.optimizer = self.raw_model.configure_optimizers(
            config.weight_decay, config.learning_rate, (config.beta1, config.beta2), config.device_type)
        if self.checkpoint is not None and self.checkpoint.optimizer_state is not None:
            self.optimizer.load_state_dict(self.checkpoint.optimizer_state)
        self.checkpoint = None
        self.model = self.raw_model
        if config.compile:
            print("compiling the model... (takes a ~minute)")
            self.model = torch.compile(self.model)
        self.model = self.ddp.wrap(self.model)
        self.scheduler = LearningRateScheduler(
            config.learning_rate, config.min_lr, config.warmup_iters, config.lr_decay_iters, config.decay_lr)
        self.wandb = None
        if config.wandb_log and self.ddp.is_master:
            import wandb
            wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=self.config_snapshot)
            self.wandb = wandb

    def build_model(self):
        cfg = self.config
        if cfg.init_from == 'scratch':
            print("Initializing a new model from scratch")
            model = JERN(JERNConfig(
                n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, block_size=cfg.block_size,
                bias=cfg.bias, vocab_size=self.loader.vocab_size(), dropout=cfg.dropout))
        elif cfg.init_from == 'resume':
            print(f"Resuming training from {cfg.out_dir}")
            self.checkpoint = Checkpoint.load(cfg.out_dir, cfg.device)
            model = self.checkpoint.build_model(dropout=cfg.dropout)
            self.iter_num = self.checkpoint.iter_num
            self.best_val_loss = self.checkpoint.best_val_loss
        else:
            raise ValueError(f"Unknown init_from: {cfg.init_from}")
        if cfg.block_size < model.config.block_size:
            model.crop_block_size(cfg.block_size)
        return model.to(cfg.device)

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ('train', 'val'):
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x, y = self.loader.get_batch(split)
                with self.ctx:
                    _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def save_checkpoint(self):
        print(f"saving checkpoint to {self.config.out_dir}")
        Checkpoint(
            self.model_args,
            self.raw_model.state_dict(),
            self.optimizer.state_dict(),
            self.iter_num,
            self.best_val_loss,
            self.config_snapshot,
        ).save(self.config.out_dir)

    def evaluate(self, lr):
        losses = self.estimate_loss()
        print(f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if self.wandb is not None:
            self.wandb.log({
                'iter': self.iter_num,
                'train/loss': losses['train'],
                'val/loss': losses['val'],
                'lr': lr,
                'mfu': self.running_mfu * 100,
            })
        if losses['val'] < self.best_val_loss or self.config.always_save_checkpoint:
            self.best_val_loss = losses['val']
            if self.iter_num > 0:
                self.save_checkpoint()

    def step(self, x, y):
        cfg = self.config
        for micro_step in range(cfg.gradient_accumulation_steps):
            if self.ddp.enabled:
                self.model.require_backward_grad_sync = micro_step == cfg.gradient_accumulation_steps - 1
            with self.ctx:
                _, loss = self.model(x, y)
                loss = loss / cfg.gradient_accumulation_steps
            x, y = self.loader.get_batch('train')
            self.scaler.scale(loss).backward()
        if cfg.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return loss, x, y

    def log(self, loss, dt):
        cfg = self.config
        lossf = loss.item() * cfg.gradient_accumulation_steps
        if self.local_iter_num >= 5:
            mfu = self.raw_model.estimate_mfu(cfg.batch_size * cfg.gradient_accumulation_steps, dt)
            self.running_mfu = mfu if self.running_mfu == -1.0 else 0.9 * self.running_mfu + 0.1 * mfu
        print(f"iter {self.iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {self.running_mfu * 100:.2f}%")

    def train(self):
        cfg = self.config
        x, y = self.loader.get_batch('train')
        t0 = time.time()
        while True:
            lr = self.scheduler.get_lr(self.iter_num)
            for group in self.optimizer.param_groups:
                group['lr'] = lr
            if self.iter_num % cfg.eval_interval == 0 and self.ddp.is_master:
                self.evaluate(lr)
            if self.iter_num == 0 and cfg.eval_only:
                break
            loss, x, y = self.step(x, y)
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % cfg.log_interval == 0 and self.ddp.is_master:
                self.log(loss, dt)
            self.iter_num += 1
            self.local_iter_num += 1
            if self.iter_num > cfg.max_iters:
                break

    def close(self):
        self.ddp.close()
