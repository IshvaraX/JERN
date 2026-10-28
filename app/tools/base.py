import os

from saturday.plugins import Plugin as BasePlugin

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


class Plugin(BasePlugin):
    """Saturday's Plugin plus what JERN needs: a sandboxed workspace, per-OS wording, web UI display flags."""

    terminal = False  # show this tool's command and output in the web terminal
    expand = False  # show this tool's result expanded (not collapsed) under the reply

    def __init__(self, config):
        super().__init__(config)
        self.workspace = Sandbox(config.path(config.workspace))
        self.parameters, self.description = self.template(config)

    @staticmethod
    def shell_name(config):
        return SHELL_NAMES.get(config.platform_name().split()[0], DEFAULT_SHELL)

    @classmethod
    def template(cls, config):
        return cls.parameters, cls.description

    @classmethod
    def describe_class(cls, config):
        # Same text as describe() on an instance, without instantiating (tools must not run on the model VM).
        parameters, description = cls.template(config)
        return f"- {cls.name}({', '.join(parameters)}): {description}"
