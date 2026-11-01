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
                                 'search_location', 'weather'})
        catalog = ToolCatalog(self.config)
        self.assertEqual(set(catalog.classes), names)
        self.assertEqual(catalog.describe(), self.registry.describe())
        catalog.close()

    def test_timeline_and_confirmation(self):
        ScriptedAssistant.replies = [
            '<tool_call>{"name": "download", "arguments": {"url": "https://example.com/report.pdf"}}</tool_call>',
            'Okay, I will not download it.',
        ]
        log = EventLog()
        assistant = ScriptedAssistant(self.config, self.registry, log)
        server = ChatServer(assistant, log)
        session = Session('abc')
        reply, call = assistant.respond(session, 'download the report')
        self.assertEqual(reply, 'I would like to download https://example.com/report.pdf. Allow?')
        self.assertIs(session.pending, call)
        response = server.response(session, reply, call)
        self.assertEqual((response.action_id, response.steps), (call.id, []))
        reply, call = assistant.resume(session, False)
        self.assertEqual((reply, call), ('Okay, I will not download it.', None))
        self.assertEqual(session.history[-2], {'role': 'tool', 'content': 'denied by user'})
        self.assertTrue(any(e['kind'] == 'wait' for e in log.since(0)))

    def test_hosted_mode_password_and_disabled_tools(self):
        from base64 import b64encode

        from fastapi.testclient import TestClient

        config = configure(['--mcp=', '--disabled_tools=open_website,play_youtube,search_location'], ROOT)
        registry = ToolRegistry(config)
        self.assertEqual({p.name for p in registry}, {'search_web', 'fetch_web', 'download', 'weather'})
        os.environ['JERN_PASSWORD'] = 'secret'
        try:
            client = TestClient(ChatServer(ScriptedAssistant(config, registry, EventLog()), EventLog()).app)
        finally:
            del os.environ['JERN_PASSWORD']
        self.assertEqual(client.get('/health').status_code, 200)
        denied = client.get('/tools')
        self.assertEqual((denied.status_code, denied.headers['www-authenticate']), (401, 'Basic realm="JERN"'))
        self.assertEqual(client.get('/tools', headers={'Authorization': 'Basic ' + b64encode(b'x:wrong').decode()}).status_code, 401)
        self.assertEqual(client.get('/tools', headers={'Authorization': 'Basic ' + b64encode(b'x:secret').decode()}).status_code, 200)
        registry.close()

    def test_sign_in_through_ishvarax_backend(self):
        from unittest import mock

        from fastapi.testclient import TestClient

        from app.server import api

        class FakeResponse:
            def __init__(self, status, body):
                self.status_code, self.body, self.ok, self.reason = status, body, status < 400, 'x'

            def json(self):
                return self.body

        def fake_post(url, json, timeout):
            if url.endswith('/auth/login'):
                if json['password'] != 'hunter22':
                    return FakeResponse(401, {'detail': 'Invalid credentials'})
                return FakeResponse(200, {'message': 'Login successful', 'token': 'backend-token',
                                          'user': {'username': 'vivek', 'email': 'v@example.com', 'photo': 'x' * 100}})
            if url.endswith('/auth/register'):
                return FakeResponse(201, {'message': 'User registered successfully'})
            raise AssertionError(url)

        ScriptedAssistant.replies = ['Hello Vivek!']
        os.environ['AUTH_URL'] = 'https://hfi-backend.example'
        try:
            server = ChatServer(ScriptedAssistant(self.config, self.registry, EventLog()), EventLog())
        finally:
            del os.environ['AUTH_URL']
        client = TestClient(server.app)
        self.assertEqual(client.get('/auth/me').status_code, 401)
        self.assertEqual(client.get('/tools').status_code, 401)
        self.assertEqual(client.post('/chat', json={'message': 'hi'}).status_code, 401)
        with mock.patch.object(api.requests, 'post', fake_post):
            bad = client.post('/auth/login', json={'username': 'vivek', 'password': 'nope'})
            self.assertEqual((bad.status_code, bad.json()['detail']), (401, 'Invalid credentials'))
            good = client.post('/auth/login', json={'username': 'v@example.com', 'password': 'hunter22'}).json()
            self.assertEqual(good['user'], {'username': 'vivek', 'email': 'v@example.com'})
            signup = client.post('/auth/register', json={'username': 'newbie', 'email': 'n@example.com', 'password': 'hunter22'})
            self.assertEqual(signup.status_code, 200)
        headers = {'Authorization': f"Bearer {good['token']}"}
        self.assertEqual(client.get('/auth/me', headers=headers).json(), {'enabled': True, 'user': 'vivek'})
        self.assertEqual(client.get('/tools', headers=headers).status_code, 200)
        chat = client.post('/chat', json={'message': 'hi'}, headers=headers).json()
        self.assertEqual(chat['reply'], 'Hello Vivek!')
        self.assertIn(('vivek', chat['session_id']), server.sessions)
        self.assertEqual(client.get('/tools', headers={'Authorization': 'Bearer ' + good['token'][:-2] + 'xx'}).status_code, 401)


if __name__ == '__main__':
    unittest.main()
