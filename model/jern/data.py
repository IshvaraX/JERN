"""Character-level dataset for the from-scratch model. Chat fine-tuning data is handled by saturday.models.data."""
import os
import pickle

import numpy as np

from constants import CONVERSATIONS_FILE, META_FILE, MIN_VAL_CHARS


class ChatDatasetPreparer:

    SOURCE = CONVERSATIONS_FILE
    MIN_VAL_CHARS = MIN_VAL_CHARS

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.source_path = os.path.join(data_dir, self.SOURCE)

    def load_text(self):
        with open(self.source_path, 'r', encoding='utf-8') as f:
            return f.read()

    def splits(self, data):
        return {'train': data, 'val': data[-max(len(data) // 10, self.MIN_VAL_CHARS):]}

    def run(self):
        data = self.load_text()
        print(f"length of dataset in characters: {len(data):,}")
        chars = sorted(set(data))
        print(f"vocab size: {len(chars):,}")
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        for split, text in self.splits(data).items():
            ids = np.array([stoi[c] for c in text], dtype=np.uint16)
            print(f"{split} has {len(ids):,} tokens")
            ids.tofile(os.path.join(self.data_dir, f'{split}.bin'))
        with open(os.path.join(self.data_dir, META_FILE), 'wb') as f:
            pickle.dump({'vocab_size': len(chars), 'itos': itos, 'stoi': stoi}, f)
