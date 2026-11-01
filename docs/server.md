# Server, API and tools

Two FastAPI servers share a base class in `app/server/api.py`:

- **`ChatServer`** (`python main.py serve`) — the assistant: web UI, chat API, sessions and tool plugins. Needs a chat backend: either the model in-process or a `ModelServer` reached through `--model_url`.
- **`ModelServer`** (`python main.py model`) — the model alone: `POST /generate` turns a message list into text. Stateless; one request at a time.

Splitting them is what lets the model run on a Colab GPU while the tools keep running on your machine: the `ChatServer` and every plugin execute locally and only the generation call leaves the machine.

## HTTP API (ChatServer, default port 8000)

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/` | | the web UI |
| `GET` | `/icon.svg` | | the logo (`app/ui/icon.svg`) |
| `POST` | `/chat` | `{"message": "...", "session_id": "..."?}` | `ChatResponse` |
| `POST` | `/confirm` | `{"session_id", "action_id", "allow": bool}` | `ChatResponse` |
| `GET` | `/tools` | | `[{name, description, confirm}]` |
| `GET` | `/auth/me` | | `{enabled, user}` — whether sign-in is configured and who the bearer token belongs to |
| `POST` | `/auth/login` | `{"username", "password"}` | `{token, user}` (username or email accepted) |
| `POST` | `/auth/register` | `{"username", "email", "password"}` | `{token, user}` — registers, then signs in |
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

Set `JERN_PASSWORD` (in `.env` or the environment) to require HTTP Basic auth (any username) on every route except `/health`. Always set it when the server is reachable from outside your machine.

### User sign-in (Ishvarax account)

Set `AUTH_URL` to the Ishvarax backend (`https://hfi-backend.vercel.app`) and the UI asks users to sign in with their Ishvarax account or create one before chatting. JERN forwards `/auth/login` and `/auth/register` to the backend's endpoints of the same name server-to-server (so no CORS changes are needed there), then issues its **own** HMAC-signed session token (`{username, exp}`, 30 days, signed with `JERN_SECRET`). `/chat`, `/confirm` and `/tools` then require `Authorization: Bearer <token>`; sessions are keyed by `(user, session_id)` so one user cannot resume another's. The backend's secret is never shared with JERN; if `JERN_SECRET` is unset a random key is generated per process and everyone signs in again after a restart. Leave `AUTH_URL` empty and the UI works without accounts, as before.

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

Tools live in their own package, deliberately outside `app/server`, because they act on the user's machine: the browser, the downloads folder, the network. Two views of the package, both satisfying Saturday's registry protocol (`describe()`, `get(name)`, iteration):

- **`ToolRegistry(config)`** — live instances (downloads sandbox). Created only by `serve`, i.e. only on the user's machine.
- **`ToolCatalog(config)`** — the tool classes and their descriptions, nothing instantiated. `saturday.toml` names it as the registry (`[plugins] registry = "app.tools:ToolCatalog"`), so `model` and `finetune` — and the Colab VM — build the system prompt with the exact tool list without ever being able to run a tool.

A tool is a class with class attributes and a `run` method; both views discover every `Plugin` subclass in the package automatically. `app.tools.base.Plugin` extends `saturday.plugins.Plugin` with the downloads sandbox and the UI flag.

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

`self.workspace` is a `Sandbox` rooted at `--workspace` for tools that save files (see `Download`); `resolve()` refuses paths that escape it. Override the classmethod `template(config)` to derive `parameters`/`description` from the configured platform. Return a string; raise `ValueError` for bad input. Results starting with `error:` are shown as errors.

Built-in tools:

| Tool | Confirm | What it does |
|---|---|---|
| `search_web` | | DuckDuckGo HTML search, top 5 results |
| `fetch_web` | | readable text of a page (3000 chars) |
| `download` | yes | save an http(s) URL into `--downloads` (50 MB max) |
| `open_website` | | open a known site name, domain or URL in the browser |
| `play_youtube` | | play the first YouTube match for a song or video |
| `search_location` | | show a place on Google Maps |
| `weather` | | current weather for a city (`OPENWEATHER_API_KEY` in `.env`) |

The last four were Saturday's automation and moved here when Saturday became a framework.

## MCP servers as tools (`app/tools/mcp.py`)

Any [Model Context Protocol](https://modelcontextprotocol.io) server can be plugged in as extra tools. Configure them in `mcp.json` at the repo root (same shape VS Code and Claude Desktop use); secrets go in `.env` (copy `.env.example`) and are referenced as `${NAME}`:

```json
{
  "servers": {
    "maps": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-google-maps"],
              "env": { "GOOGLE_MAPS_API_KEY": "${GOOGLE_MAPS_API_KEY}" },
              "tools": ["maps_search_places", "maps_directions"] },
    "figma": { "url": "http://127.0.0.1:3845/mcp", "disabled": true }
  }
}
```

- `url` → streamable-HTTP transport; `command`/`args`/`env`/`cwd` → stdio transport. Commands are resolved on `PATH` (so `npx` finds `npx.cmd` on Windows); relative paths are relative to the repo root.
- `${VAR}` is expanded from `.env` or the environment. A server whose variables are unset is **skipped with a hint** rather than started with empty credentials. `"disabled": true` skips a server outright.
- `tools` is an allowlist (glob patterns) and `exclude` a denylist. Use them: every tool description costs prompt tokens on each turn and a 1.5B model picks tools reliably only up to roughly a dozen.
- On startup every enabled server is connected (`McpClient`, one background event loop) and each of its tools becomes a `Plugin` subclass named `<server>_<tool>` (the prefix is not repeated when the tool already starts with it, so `maps_directions`, not `maps_maps_directions`). A server that fails to start is logged and skipped; the assistant still starts.
- Tool descriptions and parameter docs come from the server's JSON schema. The model writes string arguments; they are coerced to the schema's `integer` / `number` / `boolean` / `object` types before the call.
- Tools the server marks `readOnlyHint` run directly; everything else asks for approval, like `download`.
- Results (text content, or `structuredContent` as JSON) are truncated to `MCP_MAX_RESULT`.
- `--mcp=path` selects another config, `--mcp=''` disables MCP entirely. `mcp.json`, `.env` and `mcp-servers/` are **not** shipped to Colab (`ARCHIVE_EXCLUDE`): MCP servers are local processes, so they count as tools and stay on your machine.

### The configured Google servers

The shipped `mcp.json` wires five servers. Each needs a credential in `.env`; `python main.py serve` prints which ones are still missing.

| Server | Package | Credential | Where to get it |
|---|---|---|---|
| `tasks` | [arpitbatra123/mcp-googletasks](https://github.com/arpitbatra123/mcp-googletasks), cloned and built into `mcp-servers/google-tasks` | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Google Cloud Console → enable **Google Tasks API** → Credentials → OAuth client ID, type *Web application*, redirect URI `http://localhost:3000/oauth2callback`. First use: tell JERN to *authenticate*, open the URL, then give it the code (*set auth code …*). The refresh token is saved under `~/.config/google-tasks-mcp/`. |
| `keep` | [keep-mcp](https://github.com/feuerdev/keep-mcp) (installed in the venv, `venv/Scripts/keep-mcp.exe`) | `GOOGLE_EMAIL`, `GOOGLE_MASTER_TOKEN` | Keep has no public API; this uses the unofficial `gkeepapi`. Obtain a master token once with the browser flow in the [gpsoauth README](https://github.com/simon-weber/gpsoauth#alternative-flow). The token grants full account access — treat it like a password. By default the server only modifies notes it created (`UNSAFE_MODE=false`). |
| `youtube` | [zubeid-youtube-mcp-server](https://github.com/ZubeidHendricks/youtube-mcp-server) via `npx` | `YOUTUBE_API_KEY` | Cloud Console → enable **YouTube Data API v3** → API key. Free quota is 10 000 units/day; a search costs 100. |
| `google` | [@adenot/mcp-google-search](https://github.com/adenot/mcp-google-search) via `npx` | `GOOGLE_SEARCH_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID` | Cloud Console → enable **Custom Search API** → API key; then create a [Programmable Search Engine](https://programmablesearchengine.google.com) with *Search the entire web* on and copy its ID (`cx`). 100 free queries/day. The built-in `search_web` (DuckDuckGo) needs no key. |
| `maps` | [@modelcontextprotocol/server-google-maps](https://github.com/modelcontextprotocol/servers-archived/tree/main/src/google-maps) via `npx` | `GOOGLE_MAPS_API_KEY` | Cloud Console → enable Places, Directions, Distance Matrix and Geocoding APIs → API key (billing account required; generous free tier). |

To rebuild the Tasks server after a `git pull`: `cd mcp-servers/google-tasks; npm install; npm run build`. Node 20+ is required for the `npx` servers.

With all five enabled and the shipped allowlists, JERN exposes 29 MCP tools plus the 8 built-ins. That is a lot for the default 1.5B model; disable the servers you aren't using in a session, or run a 3B/7B model via `colab -- serve` when you want all of them.

## Web UI (`app/ui/index.html`)

A single HTML file served at `/`, styled with Tailwind CSS v4 (browser build from a CDN, so still no build step): light theme with one blue accent, design tokens in `@theme`, reusable component classes in `@layer components`; chat list in `localStorage`, markdown rendering, a keyboard-operable activity timeline for tool calls, inline approve/reject buttons, and a sidebar that becomes a drawer below 768 px. No build step; edit and reload.

## Hosting on Render

`render.yaml` at the repo root is a Render Blueprint for the **chat server only**: Render has no GPUs, so the model must run elsewhere and the service reaches it through `--model_url`. Tools run inside the Render container, which is why the blueprint disables the three browser tools (there is no browser to open) and MCP (its servers are local processes).

1. Run a model server somewhere reachable over HTTPS: `python main.py model --local` on a GPU box, or a Colab session exposed through a tunnel (ngrok, Cloudflare Tunnel). It must answer `GET /health` and `POST /generate`.
2. In Render: *New → Blueprint*, pick this repo. Fill in the environment variables it asks for:
   - `MODEL_URL` — the model server's base URL (required; the service exits if it cannot reach it within two minutes).
   - `AUTH_URL` — preset to the Ishvarax backend so visitors sign in with their Ishvarax account; `JERN_SECRET` is generated for you.
   - `JERN_PASSWORD` — optional extra gate in front of everything (HTTP Basic).
   - `OPENWEATHER_API_KEY` — only if you want `weather` there.
3. Deploy. The build installs `requirements-server.txt` (no torch); the start command is
   `python main.py serve --host=0.0.0.0 --port=$PORT --mcp= --model_url=$MODEL_URL --disabled_tools=open_website,play_youtube,search_location`.

Downloads land on the container's ephemeral disk, so `download` is of limited use there; the free plan sleeps after 15 minutes idle and the first request afterwards takes about a minute.
