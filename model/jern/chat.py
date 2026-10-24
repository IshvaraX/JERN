import os
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field

import requests
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import BASE_MODEL, CHAT_OUT_DIR, CONVERSATIONS_PATH, SEED
from model.jern.config import default_device, seed_everything
from model.jern.data import ConversationDataset, ConversationFile
from model.jern.util import wait_for_health


@dataclass
class ChatConfig:
    model_name: str = BASE_MODEL
    model_url: str = ''  # when set, generation is delegated to a ModelServer at this URL
    data_path: str = CONVERSATIONS_PATH
    out_dir: str = CHAT_OUT_DIR
    train: bool = False
    train_layers: int = 6
    epochs: int = 2
    learning_rate: float = 1e-4
    batch_size: int = 1
    accumulation: int = 2  # optimizer step every N batches
    max_length: int = 1024
    temperature: float = 0.3
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 6  # a phrase of this many tokens can appear only once in a reply
    max_new_tokens: int = 300
    seed: int = SEED
    device: str = field(default_factory=default_device)


class ChatBackend:

    def __init__(self, config, log=print):
        self.config = config
        self.log = log
        self.source = None

    def load(self, system_prompt=None):
        return self

    def generate(self, messages):
        raise NotImplementedError

    def timed(self, label, func):
        started = time.time()
        result = func()
        self.log('model', f"{label} in {time.time() - started:.1f}s")
        return result


class JERNChat(ChatBackend):

    def __init__(self, config, log=print):
        super().__init__(config, log)
        self.tokenizer = None
        self.model = None

    def is_trained(self):
        return os.path.exists(os.path.join(self.config.out_dir, 'config.json'))

    def load(self, system_prompt=None):
        cfg = self.config
        seed_everything(cfg.seed)
        self.source = cfg.model_name if cfg.train or not self.is_trained() else cfg.out_dir
        print(f"loading {self.source}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.source)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.source).float().to(cfg.device)
        if self.source == cfg.model_name:
            self.finetune(system_prompt or '')
        self.model.eval()
        return self

    def trainable_parameters(self):
        layers = self.config.train_layers
        if layers <= 0:
            return list(self.model.parameters())
        self.model.requires_grad_(False)
        modules = list(self.model.model.layers[-layers:]) + [self.model.model.norm]
        for module in modules:
            module.requires_grad_(True)
        return [p for module in modules for p in module.parameters()]

    def finetune(self, system_prompt):
        cfg = self.config
        conversations = ConversationFile(cfg.data_path, system_prompt).load()
        dataset = ConversationDataset(conversations, self.tokenizer, cfg.max_length)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=dataset.collate)
        params = self.trainable_parameters()
        optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)
        cuda = cfg.device.startswith('cuda')
        scaler = torch.amp.GradScaler('cuda', enabled=cuda)
        autocast = torch.autocast('cuda', dtype=torch.float16) if cuda else nullcontext()
        # Checkpointing trades compute for memory; non-reentrant so it works with frozen embeddings.
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        print(f"fine-tuning on {len(dataset)} conversations for {cfg.epochs} epochs, "
              f"{sum(p.numel() for p in params) / 1e6:.0f}M trainable parameters")
        self.model.train()
        step = 0
        for epoch in range(cfg.epochs):
            started = time.time()
            for i, batch in enumerate(loader, 1):
                batch = {k: v.to(cfg.device) for k, v in batch.items()}
                with autocast:
                    loss = self.model(**batch).loss / cfg.accumulation
                scaler.scale(loss).backward()
                if i % cfg.accumulation == 0 or i == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    print(f"epoch {epoch + 1}/{cfg.epochs} step {step}: loss {loss.item() * cfg.accumulation:.4f} "
                          f"({time.time() - started:.0f}s)")
                    started = time.time()
        self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        self.model.eval()
        self.model.requires_grad_(False)
        # Saved in fp16: half the disk and download size; load() upcasts to fp32 again.
        self.model.half().save_pretrained(cfg.out_dir)
        self.model.float()
        self.tokenizer.save_pretrained(cfg.out_dir)
        print(f"saved fine-tuned model to {cfg.out_dir}")

    def generation_kwargs(self):
        cfg = self.config
        kwargs = dict(max_new_tokens=cfg.max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
                      repetition_penalty=cfg.repetition_penalty, no_repeat_ngram_size=cfg.no_repeat_ngram_size)
        if cfg.temperature > 0:
            kwargs.update(do_sample=True, temperature=cfg.temperature, top_p=0.9)
        else:
            kwargs.update(do_sample=False)
        return kwargs

    def generate(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(self.config.device)
        self.log('model', f"generating ({inputs['input_ids'].shape[1]} prompt tokens)")
        with torch.no_grad():
            output = self.timed('generated', lambda: self.model.generate(**inputs, **self.generation_kwargs()))
        new_tokens = output[0, inputs['input_ids'].shape[1]:]
        return trim_loop(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


LOOP = re.compile(r'(.{4,80}?)(?:\1){2,}', re.DOTALL)


def trim_loop(text):
    # Small models sometimes stutter (", build 10.0, build 10.0, ..."); keep one copy and drop the tail.
    match = LOOP.search(text)
    if match is None:
        return text
    return text[:match.start() + len(match.group(1))].rstrip(' ,;:-') + '.'


class RemoteChat(ChatBackend):

    TIMEOUT = 600

    def __init__(self, config, log=print):
        super().__init__(config, log)
        self.source = config.model_url.rstrip('/')

    def load(self, system_prompt=None):
        print(f"waiting for model server at {self.source}")
        if not wait_for_health(self.source):
            raise SystemExit(f"no model server at {self.source}")
        return self

    def generate(self, messages):
        self.log('model', f"generating remotely ({len(messages)} messages)")
        response = self.timed('generated', lambda: requests.post(
            f'{self.source}/generate', json={'messages': messages}, timeout=self.TIMEOUT))
        response.raise_for_status()
        return response.json()['text']


def chat_backend(config, log=print):
    return RemoteChat(config, log) if config.model_url else JERNChat(config, log)
