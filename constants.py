
NAME = 'JERN'
FULL_NAME = 'Just An Experimental Response Network'

DATA_DIR = 'data'
DATASET = 'chat'
CONVERSATIONS_FILE = 'conversations.txt'
CONVERSATIONS_PATH = f'{DATA_DIR}/{DATASET}/{CONVERSATIONS_FILE}'
CONTACTS_PATH = f'{DATA_DIR}/contacts.json'
USER_TAG = 'User: '
ASSISTANT_TAG = f'{NAME}: '
MIN_VAL_CHARS = 512
META_FILE = 'meta.pkl'

CHAR_OUT_DIR = 'out-jern-char'
CHECKPOINT_FILE = 'ckpt.pt'
COMPILED_PREFIX = '_orig_mod.'
SEED = 1337

MAX_SESSIONS = 1000
USER_AGENT = f'Mozilla/5.0 {NAME}/1.0'
SHELL_NAMES = {'Windows': 'Windows cmd'}
DEFAULT_SHELL = 'POSIX sh'
UI_DIR = 'ui'
INDEX_FILE = 'index.html'

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

ENV_FILE = '.env'
MCP_SERVERS_DIR = 'mcp-servers'
MCP_CONNECT_TIMEOUT = 60
MCP_CALL_TIMEOUT = 120
MCP_MAX_RESULT = 6000
