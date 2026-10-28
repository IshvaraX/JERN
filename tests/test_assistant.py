"""Headless check of JERN on Saturday: a fake model drives the tool loop, timeline and confirmation flow.
Run from the repo root: python tests/test_assistant.py"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from saturday.config import configure  # noqa: E402

from app.server.api import ChatServer  # noqa: E402
from app.server.assistant import Assistant, EventLog, Session  # noqa: E402
from app.tools import ToolCatalog, ToolRegistry  # noqa: E402


class FakeChat:
    source = 'fake'

    def __init__(self, replies):
        self.replies = list(replies)

    def load(self, system_prompt=''):
        return self

    def generate(self, messages):
        return self.replies.pop(0)


class ScriptedAssistant(Assistant):
    replies = []

    def load_chat(self):
        return FakeChat(self.replies)


class JernOnSaturdayTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = configure(['--mcp='], ROOT)
        cls.registry = ToolRegistry(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.registry.close()

    def test_config_comes_from_saturday_toml(self):
        c = self.config
        self.assertEqual((c.name, c.model_name, c.out_dir), ('JERN', 'Qwen/Qwen2.5-1.5B-Instruct', 'out-jern'))
        self.assertEqual((c.workspace, c.port, c.web), ('workspace', 8000, False))
        self.assertEqual(c.registry, 'app.tools:ToolCatalog')

    def test_tools_include_the_moved_automation(self):
        names = {p.name for p in self.registry}
        for name in ('open_website', 'play_youtube', 'search_location', 'weather', 'send_email', 'shutdown',
                     'close_tab', 'connected_devices', 'run_command', 'list_files', 'system_info'):
            self.assertIn(name, names)
        catalog = ToolCatalog(self.config)
        self.assertEqual(set(catalog.classes), names)
        self.assertEqual(catalog.describe(), self.registry.describe())
        catalog.close()

    def test_rewrite_timeline_and_confirmation(self):
        ScriptedAssistant.replies = [
            '<tool_call>{"name": "run_command", "arguments": {"command": "hostname"}}</tool_call>',
            'You are on this machine.',
            '<tool_call>{"name": "shutdown", "arguments": {}}</tool_call>',
            'Okay, leaving it on.',
        ]
        log = EventLog()
        assistant = ScriptedAssistant(self.config, self.registry, log)
        server = ChatServer(assistant, log)
        session = Session('abc')
        reply, call = assistant.respond(session, 'what is my hostname')
        self.assertEqual((reply, call), ('You are on this machine.', None))
        self.assertEqual([s['kind'] for s in session.steps], ['note', 'tool'])
        self.assertEqual(session.steps[1]['name'], 'system_info')
        reply, call = assistant.respond(session, 'shut down the computer')
        self.assertEqual(reply, 'I would like to shut down the computer. Allow?')
        self.assertIs(session.pending, call)
        response = server.response(session, reply, call)
        self.assertEqual(response.action_id, call.id)
        reply, call = assistant.resume(session, False)
        self.assertEqual((reply, call), ('Okay, leaving it on.', None))
        self.assertEqual(session.history[-2], {'role': 'tool', 'content': 'denied by user'})
        self.assertTrue(any(e['kind'] == 'wait' for e in log.since(0)))


if __name__ == '__main__':
    unittest.main()
