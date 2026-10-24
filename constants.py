# Fixed values shared across JERN. Tunable hyper-parameters live in the config dataclasses instead.

# --- identity -------------------------------------------------------------------------------------
NAME = 'JERN'
FULL_NAME = 'Just An Experimental Response Network'
AUTHOR = 'Vivek Thakur'
PERSONA = (
    f'You are {NAME} ({FULL_NAME}), a small AI assistant created by {AUTHOR}. '
    'Vivek is the son of Anamika Thakur, from Jharkhand, India, and lives in Bengaluru. '
    'He is a Computer Science undergraduate at BITS Pilani and a Cloud Architect at Wipro, '
    'working onsite for HPE on private air-gapped clouds built with HPE GreenLake. '
    'His email is vivekthakurcse20509@gmail.com. '
    'Be warm, natural and helpful. Answer in a few clear sentences, use markdown lists when listing things, '
    'include links when you have them, and remember what the user tells you, such as their name.'
)
TOOL_PROMPT = (
    'You can use tools. When the user asks for something that needs one, reply with only:\n'
    '<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>\n'
    'Tools:\n{tools}\n'
    'This computer runs {platform}. '
    'The result arrives in a tool message; then answer using only that result. '
    'Do not use tools for normal conversation.'
)

# --- training data --------------------------------------------------------------------------------
DATA_DIR = 'data'
DATASET = 'chat'
CONVERSATIONS_FILE = 'conversations.txt'
CONVERSATIONS_PATH = f'{DATA_DIR}/{DATASET}/{CONVERSATIONS_FILE}'
SPEAKERS = {'User': 'user', NAME: 'assistant', 'AI': 'assistant', 'Tool': 'tool'}
SEPARATOR = '---'
USER_TAG = 'User: '
ASSISTANT_TAG = f'{NAME}: '
IGNORE_INDEX = -100
MIN_VAL_CHARS = 512
META_FILE = 'meta.pkl'

# --- models ---------------------------------------------------------------------------------------
BASE_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
CHAT_OUT_DIR = 'out-jern'  # fine-tuned instruct model (HF format)
CHAR_OUT_DIR = 'out-jern-char'  # from-scratch character-level model
CHECKPOINT_FILE = 'ckpt.pt'
COMPILED_PREFIX = '_orig_mod.'
SEED = 1337

# --- server ---------------------------------------------------------------------------------------
HOST = '127.0.0.1'
PORT = 8000
MODEL_PORT = 8001
MAX_SESSIONS = 1000
WORKSPACE_DIR = 'workspace'
DOWNLOADS_DIR = f'{WORKSPACE_DIR}/downloads'
USER_AGENT = f'Mozilla/5.0 {NAME}/1.0'
SHELL_NAMES = {'Windows': 'Windows cmd'}
DEFAULT_SHELL = 'POSIX sh'
UI_DIR = 'ui'
INDEX_FILE = 'index.html'

# --- MCP (Model Context Protocol) servers used as extra tools -------------------------------------
MCP_CONFIG = 'mcp.json'  # same shape as VS Code / Claude Desktop: {"servers": {name: {command|url, ...}}}
ENV_FILE = '.env'  # KEY=VALUE secrets referenced from mcp.json as ${KEY}
MCP_SERVERS_DIR = 'mcp-servers'  # local checkouts of MCP servers that are not published to npm/PyPI
MCP_CONNECT_TIMEOUT = 60
MCP_CALL_TIMEOUT = 120
MCP_MAX_RESULT = 6000

# --- colab ----------------------------------------------------------------------------------------
EXPORT_DIR = 'export'
LOCAL_ZIP = 'jern.zip'
REMOTE_ROOT = '/content/jern'
REMOTE_ZIP = '/content/jern.zip'
REMOTE_OUT_ZIP = '/content/jern-out.zip'
REMOTE_LOG = '/content/jern-model.log'
REMOTE_OK = '__JERN_OK__'  # printed by every remote script on success; IPython swallows SystemExit so exit codes are useless
REMOTE_STARTED = '__JERN_STARTED__'  # first line of every remote script: distinguishes our failures from the CLI's
REMOTE_TIMEOUT = 3 * 3600
TRANSIENT_ERRORS = ('WebSocketConnectionClosedException', 'socket is already closed', 'Connection is already closed',
                    'ConnectionResetError', 'TimeoutError')
RETRY_DELAY = 15
UPLOAD_RETRIES = 6  # DNS for a new runtime can lag by a minute or more
ARCHIVE_EXCLUDE = ('venv', 'env', '.git', '__pycache__', '.ipynb_checkpoints', 'out-*', EXPORT_DIR, 'docs',
                   'notebook', MCP_CONFIG, ENV_FILE, MCP_SERVERS_DIR, 'node_modules',
                   '*.bin', '*.pkl', '*.pt', '*.pyc', '*.zip')  # MCP servers and secrets are local-only
CA_BUNDLE_FILE = 'ca-bundle.pem'
SERVER_AUTH_OID = '1.3.6.1.5.5.7.3.1'
SSH_KEYS = ('id_ed25519', 'id_ecdsa')  # the only key types the Colab SSH endpoint accepts
SSH_HOST = 'root@colab'
SSH_OPTIONS = ('StrictHostKeyChecking=no', 'ExitOnForwardFailure=yes', 'ServerAliveInterval=30')
SESSION_LINE = r'^\[(\S+)\] \S+ \| '  # `colab sessions` prints "[name] endpoint | Hardware: ..." per session; "[colab]" is the CLI's log prefix
