import platform
import re
import subprocess
import threading
import time
import uuid

from app.tools.base import Plugin

WINDOWS = platform.system() == 'Windows'


class Shell:

    MAX_OUTPUT = 3000

    def __init__(self, cwd, timeout):
        self.cwd = cwd
        self.timeout = timeout
        self.lock = threading.Lock()
        self.process = None
        self.lines = []
        self.condition = threading.Condition()

    def start(self):
        if WINDOWS:
            args = ['cmd.exe', '/Q', '/K', 'prompt $_']
        else:
            args = ['/bin/sh']
        self.process = subprocess.Popen(
            args, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0)
        self.lines = []
        threading.Thread(target=self.pump, args=(self.process,), daemon=True).start()
        if not WINDOWS:
            self.process.stdin.write('PS1=""; PS2=""\n')
            self.process.stdin.flush()

    def pump(self, process):
        for line in process.stdout:
            with self.condition:
                self.lines.append(line.rstrip('\r\n'))
                self.condition.notify_all()
        with self.condition:
            self.lines.append(None)
            self.condition.notify_all()

    @property
    def alive(self):
        return self.process is not None and self.process.poll() is None

    CD = re.compile(r'^(\s*cd)\s+(?!/d\b)(?=\S)', re.IGNORECASE)

    def run(self, command):
        with self.lock:
            if not self.alive:
                self.start()
            if WINDOWS:
                command = self.CD.sub(r'\1 /d ', command)
            marker = f'__jern_{uuid.uuid4().hex}__'
            if WINDOWS:
                script = f'{command}\r\n@echo {marker} %ERRORLEVEL% %CD%\r\n'
            else:
                script = f'{command}\necho "{marker} $? $(pwd)"\n'
            with self.condition:
                self.lines.clear()
            try:
                self.process.stdin.write(script)
                self.process.stdin.flush()
            except OSError:
                self.kill()
                return f'exit code 1\nthe shell died; restarted in {self.cwd}', None
            output, code, cwd = self.collect(marker)
            if code is None:
                self.kill()
                return f'command timed out after {self.timeout} seconds; the terminal was reset', None
            self.cwd = cwd or self.cwd
            output = '\n'.join(output).strip()
            if not output:
                return f'exit code {code} (no output)', self.cwd
            return f'exit code {code}\n{output[-self.MAX_OUTPUT:]}', self.cwd

    def collect(self, marker):
        deadline = time.monotonic() + self.timeout
        output, index = [], 0
        while True:
            with self.condition:
                while index >= len(self.lines):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self.condition.wait(remaining):
                        return output, None, None
                line = self.lines[index]
                index += 1
            if line is None:
                return output, 1, None
            if marker in line:
                if 'echo' in line.split(marker)[0]:
                    continue
                _, _, rest = line.partition(marker)
                code, _, cwd = rest.strip().partition(' ')
                return output, code or '?', cwd.strip() or None
            output.append(line)

    def kill(self):
        if self.process is None:
            return
        try:
            if WINDOWS:
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.process.pid)], capture_output=True)
            else:
                self.process.kill()
        except OSError:
            pass
        self.process = None

    def close(self):
        with self.lock:
            self.kill()


class RunCommand(Plugin):

    name = 'run_command'
    required = ('command',)
    confirm = True
    terminal = True

    @classmethod
    def template(cls, config):
        shell, system = cls.shell_name(config), config.platform_name().split()[0]
        description = (f'Run a {shell} command on this {system} computer in a persistent terminal and return its '
                       f'output. The working directory and variables persist between commands, so "cd" works. '
                       f'Use only {shell} syntax; if unsure of the command, search_web first.')
        return {'command': f'the {shell} command line to run'}, description

    def __init__(self, config):
        super().__init__(config)
        self.shells = {}
        self.lock = threading.Lock()

    def summary(self, arguments):
        return f"run command: {arguments['command']}"

    def shell(self, session_id):
        with self.lock:
            if session_id not in self.shells:
                self.shells[session_id] = Shell(self.workspace.root, self.config.command_timeout)
            return self.shells[session_id]

    def cwd(self, session_id):
        shell = self.shells.get(session_id)
        return shell.cwd if shell else self.workspace.root

    def close(self, session_id):
        with self.lock:
            shell = self.shells.pop(session_id, None)
        if shell:
            shell.close()

    def close_all(self):
        for session_id in list(self.shells):
            self.close(session_id)

    def run(self, command, session_id='default'):
        output, _ = self.shell(session_id).run(command)
        return output
