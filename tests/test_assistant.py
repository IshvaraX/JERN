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

    def test_tools_are_the_web_and_browser_set(self):
        names = {p.name for p in self.registry}
        self.assertEqual(names, {'search_web', 'fetch_web', 'download', 'open_website', 'play_youtube',
                                 'search_location', 'weather', 'send_email'})
        catalog = ToolCatalog(self.config)
        self.assertEqual(set(catalog.classes), names)
        self.assertEqual(catalog.describe(), self.registry.describe())
        catalog.close()

    def test_timeline_and_confirmation(self):
        ScriptedAssistant.replies = [
            '<tool_call>{"name": "send_email", "arguments": {"contact": "mom", "subject": "hi", "body": "hello"}}</tool_call>',
            'Okay, I will not send it.',
        ]
        log = EventLog()
        assistant = ScriptedAssistant(self.config, self.registry, log)
        server = ChatServer(assistant, log)
        session = Session('abc')
        reply, call = assistant.respond(session, 'email mom saying hello')
        self.assertEqual(reply, 'I would like to email mom about "hi". Allow?')
        self.assertIs(session.pending, call)
        response = server.response(session, reply, call)
        self.assertEqual((response.action_id, response.steps), (call.id, []))
        reply, call = assistant.resume(session, False)
        self.assertEqual((reply, call), ('Okay, I will not send it.', None))
        self.assertEqual(session.history[-2], {'role': 'tool', 'content': 'denied by user'})
        self.assertTrue(any(e['kind'] == 'wait' for e in log.since(0)))


if __name__ == '__main__':
    unittest.main()
