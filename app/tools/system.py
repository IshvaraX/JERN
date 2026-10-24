import getpass
import os
import platform
import shutil
import socket
import string
import sys
import time

from app.tools.base import Plugin


class SystemInfo(Plugin):

    name = 'system_info'
    description = ('Report facts about this computer: operating system, hostname, user, CPU, architecture, '
                   'drives with free space, Python version and current folder. '
                   'Use this for any question about the device or OS instead of run_command.')
    parameters = {'topic': 'optional: os, hostname, user, cpu, drives, python, or "all"'}

    def summary(self, arguments):
        topic = arguments.get('topic') or 'all'
        return 'read system information' if topic == 'all' else f"read system information: {topic}"

    def drives(self):
        if os.name == 'nt':
            roots = [f"{letter}:\\" for letter in string.ascii_uppercase if os.path.exists(f"{letter}:\\")]
        else:
            roots = ['/']
        lines = []
        for root in roots:
            try:
                usage = shutil.disk_usage(root)
            except OSError:
                continue
            lines.append(f"{root} {usage.free / 1e9:.1f} GB free of {usage.total / 1e9:.1f} GB")
        return lines

    def run(self, topic='all'):
        uname = platform.uname()
        info = {
            'os': f"{uname.system} {uname.release} (version {uname.version})",
            'hostname': socket.gethostname(),
            'user': getpass.getuser(),
            'architecture': f"{uname.machine}, {os.cpu_count()} logical CPUs",
            'cpu': uname.processor or platform.processor() or 'unknown',
            'python': f"{sys.version.split()[0]} at {sys.executable}",
            'workspace': self.workspace.root,
            'local time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'drives': '; '.join(self.drives()) or 'unknown',
        }
        topic = (topic or 'all').strip().lower()
        wanted = {key: value for key, value in info.items() if topic in key or topic == 'all'}
        return '\n'.join(f"{key}: {value}" for key, value in (wanted or info).items())
