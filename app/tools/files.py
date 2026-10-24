import os
import re

from app.tools.base import Plugin


class ListFiles(Plugin):

    name = 'list_files'
    description = ('List files and folders in a directory. Relative paths are inside the workspace; '
                   'absolute paths such as D:/github or C:\\Users are allowed anywhere on this computer.')
    parameters = {'path': 'folder to list, "." for the workspace root'}

    def summary(self, arguments):
        return f"list files in {arguments.get('path') or '.'}"

    def resolve(self, path):
        path = path.strip() or '.'
        drive = re.fullmatch(r'([a-zA-Z])(?::|\s+drive)?[\\/]?', path)
        if os.name == 'nt' and drive and os.path.exists(root := f"{drive[1].upper()}:\\"):
            try:
                if not os.path.isdir(self.workspace.resolve(path)):
                    return root  # "d", "D:" or "d drive" mean the drive, not a workspace folder
            except ValueError:
                return root
        if os.path.isabs(path):
            return os.path.abspath(path)
        return self.workspace.resolve(path)

    def run(self, path='.'):
        folder = self.resolve(path)
        if not os.path.isdir(folder):
            raise ValueError(f"{folder} is not a folder")
        entries = sorted(os.scandir(folder), key=lambda e: (not e.is_dir(), e.name.lower()))
        lines = [f"{e.name}/" if e.is_dir() else f"{e.name} ({e.stat().st_size / 1024:.1f} KB)" for e in entries]
        return f"{folder}:\n" + ('\n'.join(lines) if lines else 'the folder is empty')


class ReadFile(Plugin):

    name = 'read_file'
    description = 'Read a text file from the workspace.'
    parameters = {'path': 'file path relative to the workspace'}
    required = ('path',)
    MAX_CHARS = 4000

    def summary(self, arguments):
        return f"read file {arguments['path']}"

    def run(self, path):
        with open(self.workspace.resolve(path), encoding='utf-8', errors='replace') as f:
            text = f.read(self.MAX_CHARS + 1)
        return text[:self.MAX_CHARS] + ('\n...(truncated)' if len(text) > self.MAX_CHARS else '')


class WriteFile(Plugin):

    name = 'write_file'
    description = 'Create or overwrite a text file in the workspace.'
    parameters = {'path': 'file path relative to the workspace', 'content': 'text to write'}
    required = ('path', 'content')
    confirm = True

    def summary(self, arguments):
        return f"write file {arguments['path']} ({len(arguments['content'])} characters)"

    def run(self, path, content):
        target = self.workspace.resolve(path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"wrote {len(content)} characters to {self.workspace.relative(target)}"
