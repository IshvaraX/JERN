import asyncio
import fnmatch
import json
import logging
import os
import re
import shutil
import threading

from app.tools.base import Plugin
from constants import ENV_FILE, MCP_CALL_TIMEOUT, MCP_CONNECT_TIMEOUT, MCP_MAX_RESULT

ENV_VAR = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


def attr(obj, *names, default=None):
    # mcp 1.x uses camelCase (inputSchema), 2.x snake_case (input_schema).
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def dotenv(path):
    # KEY=VALUE lines; keeps API keys out of mcp.json. Real environment variables win.
    values = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    values[key.strip()] = value.strip().strip('\'"')
    return {**values, **os.environ}


def expand(value, env, missing):
    # "${FIGMA_API_KEY}" inside args/env/headers/url is replaced from the environment.
    if isinstance(value, str):
        def sub(match):
            if match.group(1) not in env:
                missing.add(match.group(1))
            return env.get(match.group(1), '')
        return ENV_VAR.sub(sub, value)
    if isinstance(value, list):
        return [expand(v, env, missing) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, env, missing) for k, v in value.items()}
    return value


def load_servers(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    env = dotenv(os.path.join(os.path.dirname(os.path.abspath(path)), ENV_FILE))
    servers = {}
    for name, spec in (data.get('servers') or data.get('mcpServers') or {}).items():
        if spec.get('disabled'):
            continue
        missing = set()
        spec = expand(spec, env, missing)
        if missing:
            print(f"mcp: {name}: skipped, set {', '.join(sorted(missing))} in {ENV_FILE} or the environment")
            continue
        if 'command' in spec:
            spec['command'] = shutil.which(spec['command']) or spec['command']  # npx -> npx.cmd on Windows
        servers[name] = spec
    return servers


def allowed(spec, tool_name):
    include, exclude = spec.get('tools'), spec.get('exclude', [])
    if include and not any(fnmatch.fnmatch(tool_name, pattern) for pattern in include):
        return False
    return not any(fnmatch.fnmatch(tool_name, pattern) for pattern in exclude)


class McpClient:
    # One background event loop; every server session lives on it for the life of the process.

    def __init__(self, servers):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name='mcp').start()
        self.closing = asyncio.Event()
        self.sessions = {}
        self.tools = {}
        for name, spec in servers.items():
            try:
                future = asyncio.run_coroutine_threadsafe(self.connect(name, spec), self.loop)
                tools = [t for t in future.result(MCP_CONNECT_TIMEOUT) if allowed(spec, t.name)]
                self.tools[name] = tools
                print(f"mcp: {name}: {len(tools)} tools")
            except Exception as error:  # a server that is down should not take the assistant down
                while isinstance(error, BaseExceptionGroup) and error.exceptions:  # unwrap anyio TaskGroup noise
                    error = error.exceptions[0]
                print(f"mcp: {name}: unavailable ({type(error).__name__}: {str(error)[:200]})")

    @staticmethod
    def transport(spec):
        if 'url' in spec:
            return McpClient.http_transport(spec)
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        env = {**os.environ, **spec.get('env', {})}
        return stdio_client(StdioServerParameters(command=spec['command'], args=spec.get('args', []), env=env,
                                                  cwd=spec.get('cwd')))

    @staticmethod
    def http_transport(spec):
        try:  # mcp 2.x
            from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
            client = create_mcp_http_client(headers=spec.get('headers')) if spec.get('headers') else None
            return streamable_http_client(spec['url'], http_client=client)
        except ImportError:  # mcp 1.x
            from mcp.client.streamable_http import streamablehttp_client
            return streamablehttp_client(spec['url'], headers=spec.get('headers'))

    async def connect(self, name, spec):
        ready = self.loop.create_future()
        asyncio.ensure_future(self.serve(name, spec, ready))
        return await ready

    async def serve(self, name, spec, ready):
        from mcp import ClientSession
        try:
            async with self.transport(spec) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    tools = (await session.list_tools()).tools
                    self.sessions[name] = session
                    ready.set_result(tools)
                    await self.closing.wait()
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error if isinstance(error, Exception) else RuntimeError(str(error)))
        finally:
            self.sessions.pop(name, None)

    def call(self, server, tool, arguments):
        session = self.sessions.get(server)
        if session is None:
            raise ValueError(f"MCP server '{server}' is not connected")
        future = asyncio.run_coroutine_threadsafe(session.call_tool(tool, arguments), self.loop)
        return future.result(MCP_CALL_TIMEOUT)

    def close(self):
        self.loop.call_soon_threadsafe(self.closing.set)
        self.loop.call_soon_threadsafe(self.loop.stop)


class McpTool(Plugin):

    client = None
    server = ''
    tool = ''
    schema = {}
    expand = True

    def summary(self, arguments):
        shown = ', '.join(f"{k}={json.dumps(v)[:40]}" for k, v in arguments.items())
        return f"{self.server}: {self.tool}({shown})"

    def validate(self, arguments):
        if not isinstance(arguments, dict):
            raise ValueError('arguments must be an object')
        arguments = {key: self.coerce(key, value) for key, value in arguments.items() if key in self.parameters}
        missing = [key for key in self.required if arguments.get(key) in (None, '')]
        if missing:
            raise ValueError(f"missing arguments: {', '.join(missing)}")
        return arguments

    def coerce(self, key, value):
        # The model writes strings; MCP schemas may want numbers, booleans or JSON objects.
        kind = self.schema.get(key, {}).get('type')
        if not isinstance(value, str):
            return value
        try:
            if kind == 'integer':
                return int(value)
            if kind == 'number':
                return float(value)
            if kind == 'boolean':
                return value.strip().lower() in ('1', 'true', 'yes', 'on')
            if kind in ('object', 'array'):
                return json.loads(value)
        except (ValueError, json.JSONDecodeError):
            raise ValueError(f"{key} must be a {kind}")
        return value

    def run(self, **arguments):
        result = self.client.call(self.server, self.tool, arguments)
        parts = []
        for item in result.content or []:
            if getattr(item, 'text', None) is not None:
                parts.append(item.text)
            elif getattr(item, 'data', None) is not None:
                parts.append(f"[{attr(item, 'mimeType', 'mime_type', default='binary')} data, {len(item.data)} bytes base64]")
            else:
                parts.append(f"[{type(item).__name__}]")
        structured = attr(result, 'structuredContent', 'structured_content')
        if structured and not parts:
            parts.append(json.dumps(structured, indent=1))
        text = '\n'.join(parts).strip() or '(no output)'
        if attr(result, 'isError', 'is_error', default=False):
            text = f"error: {text}"
        return text[:MCP_MAX_RESULT] + ('\n...(truncated)' if len(text) > MCP_MAX_RESULT else '')


def describe_parameter(spec):
    text = spec.get('description') or spec.get('title') or spec.get('type', '')
    if 'enum' in spec:
        text += f" (one of: {', '.join(map(str, spec['enum']))})"
    return ' '.join(str(text).split())[:120]


def tool_name(server, tool):
    name = tool if tool.lower().startswith(server.lower()) else f'{server}_{tool}'
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def tool_classes(client):
    for server, tools in client.tools.items():
        for tool in tools:
            schema = attr(tool, 'inputSchema', 'input_schema', default={}) or {}
            properties = schema.get('properties') or {}
            annotations = getattr(tool, 'annotations', None)
            read_only = bool(annotations and attr(annotations, 'readOnlyHint', 'read_only_hint', default=False))
            description = ' '.join((tool.description or getattr(tool, 'title', None) or tool.name).split())[:300]
            yield type(f'Mcp_{tool_name(server, tool.name)}', (McpTool,), {
                'name': tool_name(server, tool.name),
                'description': description,
                'parameters': {key: describe_parameter(value) for key, value in properties.items()},
                'required': tuple(schema.get('required') or ()),
                'confirm': not read_only,
                'client': client, 'server': server, 'tool': tool.name, 'schema': properties,
            })


def mcp_tools(path):
    servers = load_servers(path)
    if not servers:
        return None, []
    # Some servers print a startup banner to stdout; the SDK logs a traceback per line and carries on.
    logging.getLogger('mcp.client.stdio').setLevel(logging.CRITICAL)
    client = McpClient(servers)
    return client, list(tool_classes(client))
