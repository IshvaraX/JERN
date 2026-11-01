# Server, API and tools

Two FastAPI servers share a base class in `app/server/api.py`:

- **`ChatServer`** (`python main.py serve`) — the assistant: web UI, chat API, sessions and the two tools. Needs a chat backend: either the model in-process or a `ModelServer` reached through `--model_url`.
- **`ModelServer`** (`python main.py model`) — the model alone: `POST /generate` turns a message list into text. Stateless; one request at a time.

Splitting them is what lets the model run on a Colab GPU while the UI and the tools keep running on your machine: only the generation call leaves it.

## HTTP API (ChatServer, default port 8000)

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/` | | the web UI |
| `GET` | `/icon.svg` | | the logo (`app/ui/icon.svg`) |
| `POST` | `/chat` | `{"message": "...", "session_id": "..."?}` | `ChatResponse` |
| `POST` | `/confirm` | `{"session_id", "action_id", "allow": bool}` | `ChatResponse` |
| `GET` | `/tools` | | `[{name, description, confirm}]` |
| `GET` | `/health` | | `{"status": "ok"}` |
| `GET` | `/docs` | | OpenAPI UI |

`ChatResponse`:

```json
{
  "reply": "text to show",
  "session_id": "...",
  "action_id": null,          // set when a tool needs approval; answer via /confirm
  "steps": [                  // what happened this turn, in order
    {"kind": "tool|note|error", "name": "search_web", "summary": "search the web for ...",
     "result": "...", "expand": false}
  ]
}
```

Sessions live in memory (`Session`: history, pending tool call, steps of the current turn), keyed by `session_id`; omit it to start a new one. The oldest session is evicted after `MAX_SESSIONS` (1000). The UI stores its own chat list in `localStorage` and replays `session_id`.

The server binds to `127.0.0.1` and has no authentication: it is meant for the person sitting at the machine. Do not expose it to a network as is.

## ModelServer API (default port 8001)

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/generate` | `{"messages": [{"role": "system|user|assistant|tool", "content": "..."}]}` | `{"text": "..."}` |
| `GET` | `/health` | | `{"status": "ok"}` |

## The assistant loop (`app/server/assistant.py`)

`Assistant` subclasses `saturday.assistant.Assistant`, which owns the loop; JERN overrides its hooks (`note`,
`confirm_question`, `give_up`). `respond(session, message)`:

1. Append the user message, trim history to `max_history` user turns.
2. Build the system prompt: the persona from `saturday.toml` + Saturday's tool prompt with the plugin list and the OS name (`config.platform_name()`), plus a context line with the current time.
3. Generate. If the reply contains no `<tool_call>`, return it.
4. Otherwise parse the call (`ToolCall.parse`, tolerant of a few alternative key names) and validate arguments against the plugin's parameter list.
5. If the plugin requires confirmation, stash it as `session.pending` and return a question with an `action_id`; `resume(session, allow)` continues after `/confirm`.
6. Else run it, append the result as a `tool` message (truncated to `max_tool_result`), and loop — at most `max_tool_calls` times per turn.

`note` records each step in `session.steps`, which the UI renders as the activity timeline.

`saturday.assistant.build_system_prompt(config, plugins)` is also used when fine-tuning so the training data carries the exact prompt the model will see at run time. `--platform` lets a Colab VM build it for the user's OS.

## Tools (`app/tools/`)

JERN ships two read-only tools; neither asks for approval:

| Tool | What it does |
|---|---|
| `search_web` | DuckDuckGo HTML search, top 5 results with snippets |
| `weather` | current weather for a city via OpenWeather (`OPENWEATHER_API_KEY` in `.env`) |

Two views of the package, both satisfying Saturday's registry protocol (`describe()`, `get(name)`, iteration):

- **`ToolRegistry(config)`** — live instances. Created only by `serve`, i.e. only on the user's machine.
- **`ToolCatalog(config)`** — the tool classes and their descriptions, nothing instantiated. `saturday.toml` names it as the registry (`[plugins] registry = "app.tools:ToolCatalog"`), so `model` and `finetune` — and the Colab VM — build the system prompt with the exact tool list without ever being able to run a tool.

A tool is a class with class attributes and a `run` method; both views discover every `Plugin` subclass in the package automatically, so adding one is just a new file:

```python
import requests

from app.tools.base import Plugin

class WordCount(Plugin):
    name = 'word_count'
    description = 'Count the words on a web page.'
    parameters = {'url': 'http(s) address of the page'}
    required = ('url',)
    confirm = False     # True: ask the user before running
    expand = False      # True: show the result expanded under the reply

    def summary(self, arguments):
        return f"count words on {arguments['url']}"

    def run(self, url):
        return f"{len(requests.get(url, timeout=10).text.split())} words"
```

Return a string; raise `ValueError` for bad input. Results starting with `error:` are shown as errors. Keep the list short: every tool description costs prompt tokens on each turn, and a small model picks tools reliably only up to roughly a dozen. Add example conversations to `data/chat/conversations.txt` and retrain (`--train`) so the model learns when to call it.

## Web UI (`app/ui/index.html`)

A single HTML file served at `/`, styled with Tailwind CSS v4 (browser build from a CDN, so no build step): light theme with one blue accent, design tokens in `@theme`, reusable component classes in `@layer components`; chat list in `localStorage`, markdown rendering, a keyboard-operable activity timeline for tool calls, inline approve/reject buttons, and a sidebar that becomes a drawer below 768 px. Edit and reload.

