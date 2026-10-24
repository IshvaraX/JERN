import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import CONVERSATIONS_FILE, IGNORE_INDEX, META_FILE, MIN_VAL_CHARS, SEPARATOR, SPEAKERS


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


class ConversationFile:

    ROLES = SPEAKERS
    SEPARATOR = SEPARATOR

    def __init__(self, path, system_prompt):
        self.path = path
        self.system_prompt = system_prompt

    def load(self):
        conversations, current = [], []
        with open(self.path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.strip() == self.SEPARATOR:
                    if current:
                        conversations.append(current)
                        current = []
                    continue
                speaker, sep, content = line.partition(': ')
                if sep and speaker in self.ROLES:
                    current.append({'role': self.ROLES[speaker], 'content': content})
                elif current:
                    current[-1]['content'] += '\n' + line
                elif line.strip():
                    raise ValueError(f"line without a speaker: {line}")
        if current:
            conversations.append(current)
        for conversation in conversations:
            for message in conversation:
                message['content'] = message['content'].strip()
        return [[{'role': 'system', 'content': self.system_prompt}] + c for c in conversations]


class ConversationDataset(Dataset):

    def __init__(self, conversations, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.examples = [self.encode(c, max_length) for c in conversations]

    def render(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return self.tokenizer(text, add_special_tokens=False)['input_ids']

    def encode(self, messages, max_length):
        input_ids, labels, previous = [], [], 0
        for i, message in enumerate(messages):
            ids = self.render(messages[:i + 1])
            delta = ids[previous:]
            previous = len(ids)
            input_ids.extend(delta)
            labels.extend(delta if message['role'] == 'assistant' else [IGNORE_INDEX] * len(delta))
        return {'input_ids': input_ids[:max_length], 'labels': labels[:max_length]}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def collate(self, batch):
        length = max(len(item['input_ids']) for item in batch)
        pad = self.tokenizer.pad_token_id
        rows = [(item['input_ids'], item['labels'], length - len(item['input_ids'])) for item in batch]
        return {
            'input_ids': torch.tensor([ids + [pad] * n for ids, _, n in rows]),
            'attention_mask': torch.tensor([[1] * len(ids) + [0] * n for ids, _, n in rows]),
            'labels': torch.tensor([labels + [IGNORE_INDEX] * n for _, labels, n in rows]),
        }
