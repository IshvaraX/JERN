# Fixed values shared across JERN. Identity, model, prompts and tunables live in saturday.toml (Saturday's Config);
# the char-level model keeps its own config dataclasses in model/jern/.

# --- identity (the assistant's name/persona come from saturday.toml; these are for the char model and docs) -----
NAME = 'JERN'
FULL_NAME = 'Just An Experimental Response Network'

# --- training data --------------------------------------------------------------------------------
DATA_DIR = 'data'
DATASET = 'chat'
CONVERSATIONS_FILE = 'conversations.txt'
CONVERSATIONS_PATH = f'{DATA_DIR}/{DATASET}/{CONVERSATIONS_FILE}'
CONTACTS_PATH = f'{DATA_DIR}/contacts.json'  # people the send_email tool can write to
USER_TAG = 'User: '
ASSISTANT_TAG = f'{NAME}: '
MIN_VAL_CHARS = 512
META_FILE = 'meta.pkl'

# --- char-level model (model/jern/train.py, sample.py) ---------------------------------------------
CHAR_OUT_DIR = 'out-jern-char'  # from-scratch character-level model
CHECKPOINT_FILE = 'ckpt.pt'
COMPILED_PREFIX = '_orig_mod.'
SEED = 1337

# --- server ---------------------------------------------------------------------------------------
MAX_SESSIONS = 1000
USER_AGENT = f'Mozilla/5.0 {NAME}/1.0'
SHELL_NAMES = {'Windows': 'Windows cmd'}
DEFAULT_SHELL = 'POSIX sh'
UI_DIR = 'ui'
INDEX_FILE = 'index.html'

# --- browser tools --------------------------------------------------------------------------------
SITES = {
    'gmail': 'https://mail.google.com/mail/u/0/#inbox',
    'maps': 'https://www.google.com/maps',
    'github': 'https://github.com',
    'stackoverflow': 'https://stackoverflow.com',
    'instagram': 'https://www.instagram.com',
    'facebook': 'https://www.facebook.com',
    'linkedin': 'https://www.linkedin.com',
    'twitter': 'https://x.com',
    'x': 'https://x.com',
    'whatsapp': 'https://web.whatsapp.com',
    'snapchat': 'https://web.snapchat.com',
    'pinterest': 'https://in.pinterest.com',
    'youtube': 'https://www.youtube.com',
}
WEATHER_URL = 'https://api.openweathermap.org/data/2.5/weather'
YOUTUBE_SEARCH_URL = 'https://www.youtube.com/results?search_query={query}'
YOUTUBE_VIDEO_URL = 'https://www.youtube.com/watch?v={video_id}'
GOOGLE_MAPS_URL = 'https://www.google.com/maps/search/{query}'

# --- MCP (Model Context Protocol) servers used as extra tools -------------------------------------
ENV_FILE = '.env'  # KEY=VALUE secrets referenced from mcp.json as ${KEY}
MCP_SERVERS_DIR = 'mcp-servers'  # local checkouts of MCP servers that are not published to npm/PyPI
MCP_CONNECT_TIMEOUT = 60
MCP_CALL_TIMEOUT = 120
MCP_MAX_RESULT = 6000
