"""JERN's assistant: Saturday's tool-call loop plus what the web UI needs (a per-turn timeline, terminal sessions,
command rewriting for the small model) and an event log the browser polls."""
import os
import re
import threading
import time
from collections import deque

from saturday.assistant import Assistant as BaseAssistant
from saturday.assistant import Session as BaseSession
from saturday.assistant import ToolCall


class EventLog:
    """Ring buffer of log lines for the UI; callable so Saturday can use it as its `log(kind, text)`."""

    def __init__(self, limit=500):
        self.entries = deque(maxlen=limit)
        self.lock = threading.Lock()
        self.counter = 0

    def add(self, kind, text):
        text = ' '.join(str(text).split())[:400]
        with self.lock:
            self.counter += 1
            self.entries.append({'id': self.counter, 'time': time.strftime('%H:%M:%S'), 'kind': kind, 'text': text})
        print(f"[{kind}] {text}", flush=True)

    __call__ = add

    def since(self, after):
        with self.lock:
            return [entry for entry in self.entries if entry['id'] > after]


class Session(BaseSession):

    def __init__(self, session_id):
        super().__init__(user=None, session_id=session_id)
        self.steps = []  # what happened during the current turn, shown as a timeline in the UI

    def begin(self):
        self.steps = []


class Assistant(BaseAssistant):

    GIVE_UP = 'I could not complete that. Could you try again?'

    # Shell commands the small model reaches for when asked about the computer; served via os/sys instead.
    SYSTEM_COMMANDS = {'systeminfo', 'hostname', 'ver', 'winver', 'uname', 'whoami', 'wmic', 'lsb_release', 'sw_vers'}
    SYSTEM_WORDS = re.compile(r'\b(os|hostname|computername|username|version|platform|system)\b', re.IGNORECASE)
    LIST_COMMANDS = {'dir', 'ls', 'tree'}

    def give_up(self, session):
        return self.GIVE_UP

    def confirm_question(self, session, plugin, call):
        return f"I would like to {plugin.summary(call.arguments)}. Allow?"

    def run_arguments(self, session, plugin):
        return {'session_id': session.id} if plugin.terminal else {}

    def note(self, session, kind, plugin, call, result=None):
        if kind == 'error':
            session.steps.append({'kind': 'error', 'name': call.name, 'summary': f"Tool call failed: {result}"})
            return
        session.steps.append({
            'kind': 'command' if plugin.terminal else 'tool', 'name': plugin.name,
            'summary': plugin.summary(call.arguments), 'command': call.arguments.get('command'),
            'result': result, 'expand': plugin.expand,
            'cwd': plugin.cwd(session.id) if plugin.terminal else None,
        })

    def rewrite(self, session, call):
        # Shell commands that only query the system are redirected to the Python-native tools.
        if call.error or call.name != 'run_command' or not isinstance(call.arguments, dict):
            return call
        words = str(call.arguments.get('command', '')).split()
        if not words:
            return call
        head, rest = words[0].lower(), words[1:]
        tail = ' '.join(rest)
        target = None
        if head in self.plugins.plugins and head != 'run_command':
            # the model wrote a tool name as if it were a command, e.g. "system_info" or "list_files D:/"
            plugin = self.plugins.plugins[head]
            target = (head, {next(iter(plugin.parameters)): tail} if rest and plugin.parameters else {})
        elif head in self.SYSTEM_COMMANDS:
            target = ('system_info', {})
        elif head in {'echo', 'type', 'cat'} and self.SYSTEM_WORDS.search(tail) and not re.search(r'[./\\]', tail):
            target = ('system_info', {})  # e.g. "type os", "echo $HOSTNAME", "echo %COMPUTERNAME%"
        elif head in self.LIST_COMMANDS:
            flags = ('/', '-') if os.name == 'nt' else ('-',)
            paths = [w for w in rest if not w.startswith(flags)]
            path = paths[0] if paths else '.'
            if not os.path.isabs(path) and not re.match(r'^[a-zA-Z]:', path):
                path = os.path.join(self.plugins.get('run_command').cwd(session.id), path)  # relative to the terminal
            target = ('list_files', {'path': os.path.normpath(path)})
        if target is None or target[0] not in self.plugins.plugins:
            return call
        rewritten = ToolCall(*target)
        self.log('tool', f"rewrote {call.raw} -> {rewritten.raw}")
        session.steps.append({'kind': 'note', 'name': rewritten.name,
                              'summary': f"Used {rewritten.name.replace('_', ' ')} instead of running `{' '.join(words)}`"})
        return rewritten