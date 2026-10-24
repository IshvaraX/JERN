import json
import os
import platform
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

from constants import DOWNLOADS_DIR, HOST, MCP_CONFIG, MODEL_PORT, PERSONA, PORT, TOOL_PROMPT, WORKSPACE_DIR
from model.jern.chat import ChatConfig, chat_backend


@dataclass
class AppConfig(ChatConfig):
    max_history: int = 5
    max_tool_calls: int = 3
    max_tool_result: int = 1500
    workspace: str = WORKSPACE_DIR
    downloads: str = DOWNLOADS_DIR
    command_timeout: int = 60
    mcp: str = MCP_CONFIG  # MCP servers config; '' disables
    system_prompt: str = PERSONA
    platform: str = ''  # OS the tools run on; defaults to this machine (set when fine-tuning elsewhere)
    host: str = HOST
    port: int = PORT
    model_port: int = MODEL_PORT
    web: bool = False

    def platform_name(self):
        return self.platform or f"{platform.system()} {platform.release()}"


class ToolCall:

    PATTERN = re.compile(r'<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)', re.DOTALL)
    NAME_KEYS = ('name', 'tool', 'function', 'tool_name')
    ARGUMENT_KEYS = ('arguments', 'parameters', 'params', 'args', 'input')

    def __init__(self, name, arguments, error=None):
        self.id = uuid.uuid4().hex
        self.name = name
        self.arguments = arguments
        self.error = error

    @property
    def raw(self):
        return f'<tool_call>{json.dumps({"name": self.name, "arguments": self.arguments})}</tool_call>'

    @staticmethod
    def loads(text):
        # Small models write Windows paths with single backslashes ("D:\"), which is invalid JSON escaping.
        for candidate in (text, text.replace('\\', '\\\\'), re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise ValueError(f"malformed tool call: {text[:200]}")

    @classmethod
    def parse(cls, text):
        match = cls.PATTERN.search(text)
        if not match:
            return None
        try:
            data = cls.loads(match.group(1))
            if not isinstance(data, dict):
                raise ValueError('tool call must be an object')
            name = next((data[key] for key in cls.NAME_KEYS if isinstance(data.get(key), str)), None)
            arguments = next((data[key] for key in cls.ARGUMENT_KEYS if isinstance(data.get(key), dict)), None)
            if name is None and len(data) == 1:
                # shorthand like {"list_files": {"path": "D:/"}}
                name, value = next(iter(data.items()))
                arguments = value if isinstance(value, dict) else None
            if arguments is None:
                arguments = {k: v for k, v in data.items() if k not in cls.NAME_KEYS + cls.ARGUMENT_KEYS}
            return cls(str(name or ''), arguments)
        except ValueError as error:
            return cls('', {}, error=str(error))


class EventLog:

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

    def since(self, after):
        with self.lock:
            return [entry for entry in self.entries if entry['id'] > after]


class Session:

    def __init__(self, session_id):
        self.id = session_id
        self.history = []
        self.pending = None
        self.steps = []  # what happened during the current turn, shown as a timeline in the UI

    def add(self, role, content):
        self.history.append({'role': role, 'content': content})

    def trim(self, max_turns):
        starts = [i for i, m in enumerate(self.history) if m['role'] == 'user']
        if len(starts) > max_turns:
            del self.history[:starts[-max_turns]]


class Assistant:

    TOOL_PROMPT = TOOL_PROMPT

    def __init__(self, config, plugins, log):
        self.config = config
        self.plugins = plugins
        self.log = log
        self.lock = threading.Lock()
        self.chat = self.load_chat(config, plugins, log)
        self.log.add('info', f"model ready ({self.chat.source}) with tools: {', '.join(p.name for p in plugins)}")

    @classmethod
    def build_system_prompt(cls, config, plugins):
        prompt = cls.TOOL_PROMPT.replace('{tools}', plugins.describe()).replace('{platform}', config.platform_name())
        return f"{config.system_prompt}\n{prompt}"

    @classmethod
    def load_chat(cls, config, plugins, log):
        return chat_backend(config, log.add).load(cls.build_system_prompt(config, plugins))

    def system_prompt(self):
        return self.build_system_prompt(self.config, self.plugins)

    def generate(self, history):
        return self.chat.generate([{'role': 'system', 'content': self.system_prompt()}] + history)

    def respond(self, session, message):
        with self.lock:
            session.steps = []
            self.log.add('user', message)
            if session.pending is not None:
                session.pending = None
                self.record(session, 'denied by user')
            session.add('user', message)
            session.trim(self.config.max_history)
            return self.step(session)

    def resume(self, session, allow):
        with self.lock:
            session.steps = []
            call, session.pending = session.pending, None
            self.log.add('user', f"{'allowed' if allow else 'denied'}: {call.name}")
            if allow:
                self.execute(session, self.plugins.get(call.name), call)
            else:
                self.record(session, 'denied by user')
            return self.step(session)

    # Shell commands the small model reaches for when asked about the computer; served via os/sys instead.
    SYSTEM_COMMANDS = {'systeminfo', 'hostname', 'ver', 'winver', 'uname', 'whoami', 'wmic', 'lsb_release', 'sw_vers'}
    SYSTEM_WORDS = re.compile(r'\b(os|hostname|computername|username|version|platform|system)\b', re.IGNORECASE)
    LIST_COMMANDS = {'dir', 'ls', 'tree'}

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
        self.log.add('tool', f"rewrote {call.raw} -> {rewritten.raw}")
        session.steps.append({'kind': 'note', 'name': rewritten.name,
                              'summary': f"Used {rewritten.name.replace('_', ' ')} instead of running `{' '.join(words)}`"})
        return rewritten

    def step(self, session):
        for _ in range(self.config.max_tool_calls):
            answer = self.generate(session.history)
            call = ToolCall.parse(answer)
            if call is None:
                session.add('assistant', answer)
                self.log.add('ai', answer)
                return answer, None
            call = self.rewrite(session, call)
            session.add('assistant', call.raw)
            self.log.add('tool', f"call {call.raw}")
            try:
                if call.error:
                    raise ValueError(call.error)
                plugin = self.plugins.get(call.name)
                call.arguments = plugin.validate(call.arguments)
            except ValueError as error:
                self.log.add('error', f"{error} | model wrote: {answer[:300]}")
                session.steps.append({'kind': 'error', 'name': call.name, 'summary': f"Tool call failed: {error}"})
                self.record(session, f"error: {error}")
                continue
            if plugin.confirm:
                session.pending = call
                self.log.add('wait', f"waiting for permission: {plugin.summary(call.arguments)}")
                return f"I would like to {plugin.summary(call.arguments)}. Allow?", call
            self.execute(session, plugin, call)
        answer = 'I could not complete that. Could you try again?'
        session.add('assistant', answer)
        self.log.add('error', f"gave up after {self.config.max_tool_calls} tool calls")
        return answer, None

    def execute(self, session, plugin, call):
        self.log.add('tool', f"running {plugin.summary(call.arguments)}")
        try:
            extra = {'session_id': session.id} if plugin.terminal else {}
            result = plugin.run(**call.arguments, **extra)
        except Exception as error:
            result = f"error: {type(error).__name__}: {str(error)[:200]}"
        self.log.add('error' if result.startswith('error:') else 'tool', f"result: {result}")
        session.steps.append({
            'kind': 'command' if plugin.terminal else 'tool', 'name': plugin.name,
            'summary': plugin.summary(call.arguments), 'command': call.arguments.get('command'),
            'result': result, 'expand': plugin.expand,
            'cwd': plugin.cwd(session.id) if plugin.terminal else None,
        })
        self.record(session, result)

    def record(self, session, result):
        session.add('tool', result[:self.config.max_tool_result])
