"""JERN's assistant: Saturday's tool-call loop plus what the web UI needs (a per-turn timeline) and an event log the
browser polls."""
import threading
import time
from collections import deque

from saturday.assistant import Assistant as BaseAssistant
from saturday.assistant import Session as BaseSession


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
        self.steps = []

    def begin(self):
        self.steps = []


class Assistant(BaseAssistant):

    GIVE_UP = 'I could not complete that. Could you try again?'

    def give_up(self, session):
        return self.GIVE_UP

    def confirm_question(self, session, plugin, call):
        return f"I would like to {plugin.summary(call.arguments)}. Allow?"

    def note(self, session, kind, plugin, call, result=None):
        if kind == 'error':
            session.steps.append({'kind': 'error', 'name': call.name, 'summary': f"Tool call failed: {result}"})
            return
        session.steps.append({'kind': 'tool', 'name': plugin.name, 'summary': plugin.summary(call.arguments),
                              'result': result, 'expand': plugin.expand})
