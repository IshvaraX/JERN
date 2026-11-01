import os
import threading
import uuid
import webbrowser

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.server.assistant import Session
from constants import ICON_FILE, INDEX_FILE, MAX_SESSIONS, UI_DIR
from model.jern.util import wait_for_port


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)


class ConfirmRequest(BaseModel):
    session_id: str = Field(max_length=64)
    action_id: str = Field(max_length=64)
    allow: bool


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


class ChatServer(Server):

    MAX_SESSIONS = MAX_SESSIONS
    UI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), UI_DIR)

    def __init__(self, assistant, log):
        super().__init__(assistant.config.name)
        self.assistant = assistant
        self.log = log
        self.sessions = {}
        self.app.get('/', include_in_schema=False)(self.index)
        self.app.get(f'/{ICON_FILE}', include_in_schema=False)(self.icon)
        self.app.post('/chat', response_model=ChatResponse)(self.chat)
        self.app.post('/confirm', response_model=ChatResponse)(self.confirm)
        self.app.get('/tools')(self.tools)

    def banner(self, host, port):
        return (f"web ui: http://{host}:{port}/   chat endpoint: POST http://{host}:{port}/chat  "
                f"{{\"message\": \"hello\"}}   {super().banner(host, port)}")

    def index(self):
        return FileResponse(os.path.join(self.UI, INDEX_FILE), media_type='text/html')

    def icon(self):
        return FileResponse(os.path.join(self.UI, ICON_FILE), media_type='image/svg+xml')

    def tools(self):
        return [{'name': p.name, 'description': p.description, 'confirm': p.confirm} for p in self.assistant.plugins]

    def response(self, session, reply, call):
        return ChatResponse(reply=reply, session_id=session.id, action_id=call.id if call else None, steps=session.steps)

    def session(self, session_id):
        session_id = session_id or uuid.uuid4().hex
        if session_id not in self.sessions and len(self.sessions) >= self.MAX_SESSIONS:
            self.sessions.pop(next(iter(self.sessions)))
        return self.sessions.setdefault(session_id, Session(session_id))

    def close(self):
        self.assistant.plugins.close()

    def chat(self, request: ChatRequest):
        session = self.session(request.session_id)
        reply, call = self.assistant.respond(session, request.message)
        return self.response(session, reply, call)

    def confirm(self, request: ConfirmRequest):
        session = self.sessions.get(request.session_id)
        if session is None or session.pending is None or session.pending.id != request.action_id:
            raise HTTPException(status_code=404, detail='no such pending action')
        reply, call = self.assistant.resume(session, request.allow)
        return self.response(session, reply, call)
