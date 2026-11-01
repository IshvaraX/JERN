from saturday.plugins import Plugin as BasePlugin


class Plugin(BasePlugin):
    """Saturday's Plugin plus the web UI display flag."""

    expand = False

    def __init__(self, config):
        super().__init__(config)
        self.parameters, self.description = self.template(config)

    @classmethod
    def template(cls, config):
        return cls.parameters, cls.description

    @classmethod
    def describe_class(cls, config):
        parameters, description = cls.template(config)
        return f"- {cls.name}({', '.join(parameters)}): {description}"
