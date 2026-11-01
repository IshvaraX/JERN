import importlib

_EXPORTS = {
    'model.jern.config': ['Configurator', 'RuntimeConfig', 'configure', 'default_device', 'default_dtype', 'seed_everything'],
    'model.jern.network': ['JERN', 'JERNConfig', 'Checkpoint'],
}
_MODULE_OF = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = sorted(_MODULE_OF)


def __getattr__(name):
    # Lazy so that model.jern.util can be imported without torch installed.
    if name in _MODULE_OF:
        return getattr(importlib.import_module(_MODULE_OF[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
