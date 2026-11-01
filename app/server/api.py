import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import uuid
import webbrowser

import requests
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app.server.assistant import Session
from constants import (AUTH_SECRET_ENV, AUTH_TIMEOUT, AUTH_URL_ENV, ICON_FILE, INDEX_FILE, MAX_SESSIONS, PASSWORD_ENV,
                       SESSION_TTL, UI_DIR)
from model.jern.util import wait_for_port


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)


class ConfirmRequest(BaseModel):
    session_id: str = Field(max_length=64)
    action_id: str = Field(max_length=64)
    allow: bool


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=128)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r'^[A-Za-z0-9_.-]+$')
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class Step(BaseModel):
    kind: str
    name: str
    summary: str
    result: str | None = None
    expand: bool = False


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    action_id: str | None = None
    steps: list[Step] = []


class Server:

    def __init__(self, title):
        self.app = FastAPI(title=title)
        self.app.get('/health')(self.health)

    def health(self):
        return {'status': 'ok'}

    def open_browser(self, host, port):
        if wait_for_port(host, port, interval=0.5):
            webbrowser.open(f'http://{host}:{port}/')

    def banner(self, host, port):
        return f"docs: http://{host}:{port}/docs"

    def run(self, host, port, web=False):
        browse_host = '127.0.0.1' if host == '0.0.0.0' else host
        print(self.banner(browse_host, port))
        if web:
            threading.Thread(target=self.open_browser, args=(browse_host, port), daemon=True).start()
        try:
            uvicorn.run(self.app, host=host, port=port)
        finally:
            self.close()

    def close(self):
        pass


class Auth:
    """Sign-in through the Ishvarax backend's /auth/login and /auth/register; JERN then issues its own HMAC-signed
    session token so later requests are checked locally without calling the backend again."""

    def __init__(self, url, secret, ttl=SESSION_TTL):
        self.url = url.rstrip('/')
        self.key = (secret or secrets.token_hex(32)).encode('utf-8')
        self.ttl = ttl

    @property
    def enabled(self):
        return bool(self.url)

    def forward(self, path, body):
        try:
            response = requests.post(f'{self.url}/auth/{path}', json=body, timeout=AUTH_TIMEOUT)
        except requests.RequestException:
            raise HTTPException(status_code=502, detail='The sign-in service is unreachable. Try again later.')
        if response.ok:
            return response.json()
        try:
            detail = response.json().get('detail') or response.reason
        except ValueError:
            detail = response.reason
        raise HTTPException(status_code=response.status_code, detail=str(detail))

    def issue(self, username):
        payload = _b64e(json.dumps({'u': username, 'exp': int(time.time()) + self.ttl}).encode('utf-8'))
        return f"{payload}.{_b64e(hmac.new(self.key, payload.encode('utf-8'), hashlib.sha256).digest())}"

    def verify(self, token):
        try:
            payload, signature = token.split('.', 1)
            expected = hmac.new(self.key, payload.encode('utf-8'), hashlib.sha256).digest()
            if not hmac.compare_digest(_b64d(signature), expected):
                return None
            data = json.loads(_b64d(payload))
        except (ValueError, TypeError, UnicodeDecodeError):
            return None
        if not isinstance(data, dict) or data.get('exp', 0) < time.time():
            return None
        return data.get('u') or None


def _b64e(raw):
    return base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')


def _b64d(text):
    return base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))


class ChatServer(Server):

    MAX_SESSIONS = MAX_SESSIONS
    UI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), UI_DIR)

    def __init__(self, assistant, log):
        super().__init__(assistant.config.name)
        self.assistant = assistant
        self.log = log
        self.sessions = {}
        self.password = os.environ.get(PASSWORD_ENV, '')
        self.auth = Auth(os.environ.get(AUTH_URL_ENV, ''), os.environ.get(AUTH_SECRET_ENV, ''))
        if self.password:
            self.app.middleware('http')(self.require_password)
        self.app.get('/', include_in_schema=False)(self.index)
        self.app.get(f'/{ICON_FILE}', include_in_schema=False)(self.icon)
        self.app.get('/favicon.ico', include_in_schema=False)(self.icon)
        self.app.post('/chat', response_model=ChatResponse)(self.chat)
        self.app.post('/confirm', response_model=ChatResponse)(self.confirm)
        self.app.get('/tools')(self.tools)
        self.app.get('/auth/me')(self.me)
        self.app.post('/auth/login')(self.login)
        self.app.post('/auth/register')(self.register)

    async def require_password(self, request, call_next):
        if request.url.path == '/health' or self.authorized(request.headers.get('authorization', '')):
            return await call_next(request)
        return Response(status_code=401, headers={'WWW-Authenticate': f'Basic realm="{self.assistant.config.name}"'})

    def authorized(self, header):
        scheme, _, token = header.partition(' ')
        if scheme.lower() != 'basic':
            return False
        try:
            _, _, password = base64.b64decode(token).decode('utf-8').partition(':')
        except (ValueError, UnicodeDecodeError):
            return False
        return secrets.compare_digest(password, self.password)

    def banner(self, host, port):
        return (f"web ui: http://{host}:{port}/   chat endpoint: POST http://{host}:{port}/chat  "
                f"{{\"message\": \"hello\"}}   {super().banner(host, port)}")

    def index(self):
        return FileResponse(os.path.join(self.UI, INDEX_FILE), media_type='text/html')

    def icon(self):
        return FileResponse(os.path.join(self.UI, ICON_FILE), media_type='image/svg+xml')

    def user(self, authorization):
        """Username behind a `Bearer` token, or None when sign-in is not configured."""
        if not self.auth.enabled:
            return None
        scheme, _, token = (authorization or '').partition(' ')
        username = self.auth.verify(token.strip()) if scheme.lower() == 'bearer' else None
        if not username:
            raise HTTPException(status_code=401, detail='Please sign in')
        return username

    def me(self, authorization: str | None = Header(default=None)):
        if not self.auth.enabled:
            return {'enabled': False, 'user': None}
        return {'enabled': True, 'user': self.user(authorization)}

    def login(self, request: LoginRequest):
        if not self.auth.enabled:
            raise HTTPException(status_code=404, detail='sign-in is not configured')
        data = self.auth.forward('login', {'username': request.username, 'password': request.password})
        profile = data.get('user') or {}
        username = profile.get('username') or request.username
        self.log('user', f"signed in: {username}")
        return {'token': self.auth.issue(username), 'user': {'username': username, 'email': profile.get('email')}}

    def register(self, request: RegisterRequest):
        if not self.auth.enabled:
            raise HTTPException(status_code=404, detail='sign-in is not configured')
        self.auth.forward('register', {'username': request.username, 'email': request.email, 'password': request.password})
        return self.login(LoginRequest(username=request.username, password=request.password))

    def tools(self, authorization: str | None = Header(default=None)):
        self.user(authorization)
        return [{'name': p.name, 'description': p.description, 'confirm': p.confirm} for p in self.assistant.plugins]

    def response(self, session, reply, call):
        return ChatResponse(reply=reply, session_id=session.id, action_id=call.id if call else None, steps=session.steps)

    def session(self, session_id, user):
        session_id = session_id or uuid.uuid4().hex
        key = (user, session_id)
        if key not in self.sessions and len(self.sessions) >= self.MAX_SESSIONS:
            self.sessions.pop(next(iter(self.sessions)))
        return self.sessions.setdefault(key, Session(session_id))

    def close(self):
        self.assistant.plugins.close()

    def chat(self, request: ChatRequest, authorization: str | None = Header(default=None)):
        session = self.session(request.session_id, self.user(authorization))
        reply, call = self.assistant.respond(session, request.message)
        return self.response(session, reply, call)

    def confirm(self, request: ConfirmRequest, authorization: str | None = Header(default=None)):
        session = self.sessions.get((self.user(authorization), request.session_id))
        if session is None or session.pending is None or session.pending.id != request.action_id:
            raise HTTPException(status_code=404, detail='no such pending action')
        reply, call = self.assistant.resume(session, request.allow)
        return self.response(session, reply, call)
