import os

from saturday.plugins import Plugin as BasePlugin


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
    """Saturday's Plugin plus what JERN needs: a sandboxed downloads workspace and the web UI display flag."""

    expand = False

    def __init__(self, config):
        super().__init__(config)
        self.workspace = Sandbox(config.path(config.workspace))
        self.parameters, self.description = self.template(config)

    @classmethod
    def template(cls, config):
        return cls.parameters, cls.description

    @classmethod
    def describe_class(cls, config):
        parameters, description = cls.template(config)
        return f"- {cls.name}({', '.join(parameters)}): {description}"
