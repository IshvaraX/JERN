import sys
from ast import literal_eval
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields

import torch


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def default_dtype():
    return 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def configure(config_class, argv=None):
    return Configurator(config_class(), argv).apply()


@dataclass
class RuntimeConfig:
    device: str = field(default_factory=default_device)
    dtype: str = field(default_factory=default_dtype)
    compile: bool = False

    @property
    def device_type(self):
        return 'cuda' if 'cuda' in self.device else 'cpu'

    @property
    def torch_dtype(self):
        return {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]

    def autocast(self):
        if self.device_type == 'cpu':
            return nullcontext()
        return torch.amp.autocast(device_type=self.device_type, dtype=self.torch_dtype)

    def to_dict(self):
        return asdict(self)


class Configurator:

    def __init__(self, config, argv=None):
        self.config = config
        self.argv = sys.argv[1:] if argv is None else argv
        self.keys = {f.name for f in fields(config)}

    def apply(self):
        for arg in self.argv:
            if arg.startswith('--'):
                self.apply_override(arg)
            else:
                self.apply_file(arg)
        return self.config

    def apply_file(self, path):
        print(f"Overriding config with {path}")
        namespace = {}
        with open(path, encoding='utf-8') as f:
            exec(f.read(), namespace)
        for key in self.keys:
            if key in namespace:
                setattr(self.config, key, namespace[key])

    def apply_override(self, arg):
        key, has_value, raw = arg[2:].partition('=')
        if key not in self.keys:
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(self.config, key)
        if not has_value and isinstance(current, bool):
            value = True
        else:
            try:
                value = literal_eval(raw)
            except (SyntaxError, ValueError):
                value = raw
        if type(value) is not type(current):
            raise TypeError(f"{key} expects {type(current).__name__}, got {type(value).__name__}")
        print(f"Overriding: {key} = {value}")
        setattr(self.config, key, value)
