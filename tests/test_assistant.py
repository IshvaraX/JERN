"""Headless check of JERN on Saturday: a fake model drives the tool loop, timeline and confirmation flow.
Run from the repo root: python tests/test_assistant.py"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # noqa: E402
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
        cls.config = configure([], ROOT)
        cls.registry = ToolRegistry(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.registry.close()

    def test_config_comes_from_saturday_toml(self):
        c = self.config
        self.assertEqual((c.name, c.model_name, c.out_dir), ('JERN', 'microsoft/Phi-3.5-mini-instruct', 'out-jern'))
        self.assertEqual((c.port, c.web), (8000, False))
        self.assertEqual(c.registry, 'app.tools:ToolCatalog')

    def test_tools_are_search_and_weather(self):
        names = {p.name for p in self.registry}
        self.assertEqual(names, {'search_web', 'weather'})
        catalog = ToolCatalog(self.config)
        self.assertEqual(set(catalog.classes), names)
        self.assertEqual(catalog.describe(), self.registry.describe())
        catalog.close()

    def test_tool_step_lands_in_the_timeline(self):
        ScriptedAssistant.replies = [
            '<tool_call>{"name": "weather", "arguments": {"city": "Bengaluru"}}</tool_call>',
            'It is 27°C and cloudy in Bengaluru.',
        ]
        log = EventLog()
        assistant = ScriptedAssistant(self.config, self.registry, log)
        session = Session('abc')
        reply, call = assistant.respond(session, 'weather in bengaluru')
        self.assertEqual((reply, call), ('It is 27°C and cloudy in Bengaluru.', None))
        self.assertEqual([(s['kind'], s['name']) for s in session.steps], [('tool', 'weather')])
        self.assertIn('OPENWEATHER_API_KEY', session.steps[0]['result'])

    def test_http_api(self):
        ScriptedAssistant.replies = ['Hello!']
        log = EventLog()
        client = TestClient(ChatServer(ScriptedAssistant(self.config, self.registry, log), log).app)
        self.assertEqual(client.get('/health').json(), {'status': 'ok'})
        self.assertEqual(client.get('/').status_code, 200)
        self.assertEqual(client.get('/icon.svg').headers['content-type'], 'image/svg+xml')
        self.assertEqual([t['name'] for t in client.get('/tools').json()], ['search_web', 'weather'])
        data = client.post('/chat', json={'message': 'hi'}).json()
        self.assertEqual((data['reply'], data['action_id'], data['steps']), ('Hello!', None, []))
        self.assertEqual(client.post('/confirm', json={'session_id': data['session_id'], 'action_id': 'x', 'allow': True}).status_code, 404)


if __name__ == '__main__':
    unittest.main()
