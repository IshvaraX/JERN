import importlib
import pkgutil

from app.tools.base import Plugin

BUILTIN_SKIP = ('base', 'mcp')  # not tool modules


def discover():
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name in BUILTIN_SKIP:
            continue
        module = importlib.import_module(f'{__name__}.{module_info.name}')
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, Plugin) and obj.name and obj.__module__ == module.__name__:
                yield obj


class ToolCatalog:
    # Descriptions only: what the model is told about the tools. Safe to build where tools must not run (the model VM).

    def __init__(self, config):
        from app.tools.mcp import mcp_tools

        self.config = config
        self.classes = {cls.name: cls for cls in discover()}
        self.mcp, external = mcp_tools(config.mcp)
        self.classes.update({cls.name: cls for cls in external})

    def __iter__(self):
        return iter(self.classes.values())

    def describe(self):
        return '\n'.join(cls.describe(self.config) for cls in self)

    def close(self):
        if self.mcp is not None:
            self.mcp.close()


class ToolRegistry(ToolCatalog):
    # Live tool instances: sandboxes, shells, network. Only ever created on the user's machine.

    def __init__(self, config):
        super().__init__(config)
        self.plugins = {name: cls(config) for name, cls in self.classes.items()}

    def __iter__(self):
        return iter(self.plugins.values())

    def __len__(self):
        return len(self.plugins)

    def get(self, name):
        if name not in self.plugins:
            raise ValueError(f"unknown tool: {name}")
        return self.plugins[name]
