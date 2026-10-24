import json
import os

from constants import DEFAULT_SHELL, SHELL_NAMES


class Sandbox:

    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def resolve(self, path):
        full = os.path.abspath(os.path.join(self.root, path))
        if os.path.commonpath([full, self.root]) != self.root:
            raise ValueError('path is outside the workspace')
        return full

    def relative(self, full):
        return os.path.relpath(full, self.root).replace(os.sep, '/')


class Plugin:

    name = ''
    description = ''
    parameters = {}
    required = ()
    confirm = False
    terminal = False  # show this tool's command and output in the web terminal
    expand = False  # show this tool's result expanded (not collapsed) under the reply

    def __init__(self, config):
        self.config = config
        self.workspace = Sandbox(config.workspace)
        self.parameters, self.description = self.template(config)

    @staticmethod
    def shell_name(config):
        return SHELL_NAMES.get(config.platform_name().split()[0], DEFAULT_SHELL)

    @classmethod
    def template(cls, config):
        return cls.parameters, cls.description

    @classmethod
    def describe(cls, config):
        parameters, description = cls.template(config)
        return f"- {cls.name}({', '.join(parameters)}): {description}"

    def summary(self, arguments):
        return f"{self.name} {json.dumps(arguments)}"

    def validate(self, arguments):
        if not isinstance(arguments, dict):
            raise ValueError('arguments must be an object')
        # Small models invent argument names; keep the known ones rather than failing the whole call.
        arguments = {key: value for key, value in arguments.items() if key in self.parameters}
        missing = [key for key in self.required if not arguments.get(key)]
        if missing:
            raise ValueError(f"missing arguments: {', '.join(missing)}")
        if not all(isinstance(value, str) for value in arguments.values()):
            raise ValueError('arguments must be strings')
        return arguments

    def run(self, **arguments):
        raise NotImplementedError
