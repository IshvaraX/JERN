"""JERN's tool registry: every Plugin subclass in this package plus the MCP servers from mcp.json.

Both classes satisfy Saturday's registry protocol (describe(), get(name), iteration), so saturday.toml points at
them: `[plugins] registry = "app.tools:ToolCatalog"` builds descriptions only, which is what the model VM and
fine-tuning need; main.py builds a ToolRegistry (live instances) on the user's machine.
"""
import importlib
import pkgutil

from app.tools.base import Plugin

BUILTIN_SKIP = ('base', 'mcp')


def discover():
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name in BUILTIN_SKIP:
            continue
        module = importlib.import_module(f'{__name__}.{module_info.name}')
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, Plugin) and obj.name and obj.__module__ == module.__name__:
                yield obj


class ToolCatalog:

    def __init__(self, config):
        from app.tools.mcp import mcp_tools

        self.config = config
        self.classes = {cls.name: cls for cls in discover()}
        self.mcp, external = mcp_tools(config.path(config.mcp) if config.mcp else '')
        self.classes.update({cls.name: cls for cls in external})

    def __iter__(self):
        return iter(self.classes.values())

    def __len__(self):
        return len(self.classes)

    def describe(self):
        return '\n'.join(cls.describe_class(self.config) for cls in self.classes.values())

    def get(self, name):
        raise ValueError(f"tools do not run here (catalog only): {name}")

    def close(self):
        if self.mcp is not None:
            self.mcp.close()


class ToolRegistry(ToolCatalog):

    def __init__(self, config):
        super().__init__(config)
        self.plugins = {name: cls(config) for name, cls in self.classes.items()}

    def __iter__(self):
        return iter(self.plugins.values())

    def __len__(self):
        return len(self.plugins)

    def describe(self):
        return '\n'.join(plugin.describe() for plugin in self.plugins.values())

    def get(self, name):
        if name not in self.plugins:
            raise ValueError(f"unknown tool: {name}")
        return self.plugins[name]
