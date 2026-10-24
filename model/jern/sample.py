import os
import pickle
from dataclasses import dataclass

import torch

from constants import ASSISTANT_TAG, CHAR_OUT_DIR, DATA_DIR, META_FILE, SEED, USER_TAG
from model.jern.config import RuntimeConfig, seed_everything
from model.jern.network import Checkpoint


@dataclass
class SampleConfig(RuntimeConfig):
    out_dir: str = CHAR_OUT_DIR
    data_dir: str = DATA_DIR
    start: str = "\n"
    num_samples: int = 10
    max_new_tokens: int = 500
    temperature: float = 0.3
    top_k: int = 200
    seed: int = SEED
    interactive: bool = True


class CharTokenizer:

    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def from_meta(cls, path):
        with open(path, 'rb') as f:
            meta = pickle.load(f)
        return cls(meta['stoi'], meta['itos'])

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)


class Sampler:

    def __init__(self, config):
        self.config = config
        seed_everything(config.seed)
        self.model, dataset = self.load_model()
        self.tokenizer = self.load_tokenizer(dataset)

    def load_model(self):
        cfg = self.config
        checkpoint = Checkpoint.load(cfg.out_dir, cfg.device)
        model = checkpoint.build_model()
        model.eval()
        model.to(cfg.device)
        if cfg.compile:
            model = torch.compile(model)
        return model, checkpoint.config['dataset']

    def load_tokenizer(self, dataset):
        return CharTokenizer.from_meta(os.path.join(self.config.data_dir, dataset, META_FILE))

    def chat(self):
        return Chat(self.model, self.tokenizer, self.config)

    def prompt(self):
        start = self.config.start
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                return f.read()
        return start

    def run(self):
        cfg = self.config
        ids = self.tokenizer.encode(self.prompt())
        x = torch.tensor(ids, dtype=torch.long, device=cfg.device)[None, ...]
        with torch.no_grad(), cfg.autocast():
            for _ in range(cfg.num_samples):
                y = self.model.generate(x, cfg.max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k)
                print(self.tokenizer.decode(y[0].tolist()))
                print('---------------')


class Chat:

    USER = USER_TAG
    AI = ASSISTANT_TAG

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.history = ''

    def reply(self, message):
        cfg = self.config
        self.history += f"{self.USER}{message}\n{self.AI}"
        x = torch.tensor(self.tokenizer.encode(self.history), dtype=torch.long, device=cfg.device)[None, ...]
        start = x.size(1)
        text = ''
        with torch.no_grad(), cfg.autocast():
            for _ in range(cfg.max_new_tokens):
                x = self.model.generate(x, 1, temperature=cfg.temperature, top_k=cfg.top_k)
                text = self.tokenizer.decode(x[0, start:].tolist())
                if '\n' in text.lstrip():
                    break
        answer = text.strip().split('\n')[0]
        self.history += f"{answer}\n"
        return answer

    def run(self):
        print("Type a message and press Enter. Type 'exit' or press Ctrl+C to quit.")
        while True:
            try:
                message = input('You: ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if message.lower() in ('exit', 'quit'):
                break
            if message:
                print(f"{self.AI}{self.reply(message)}")
