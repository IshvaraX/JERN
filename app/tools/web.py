import re
from html.parser import HTMLParser

import requests

from app.tools.base import Plugin
from constants import USER_AGENT


class TextExtractor(HTMLParser):

    SKIP = {'script', 'style', 'noscript', 'svg'}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if not self.depth:
            self.parts.append(data)

    def text(self):
        return re.sub(r'\s+', ' ', ' '.join(self.parts)).strip()


class FetchWeb(Plugin):

    name = 'fetch_web'
    description = "Fetch a web page's readable text to answer questions about it."
    parameters = {'url': 'http(s) URL of the page'}
    required = ('url',)
    MAX_BYTES = 1024 * 1024
    MAX_CHARS = 3000

    def summary(self, arguments):
        return f"read web page {arguments['url']}"

    def run(self, url):
        if not url.startswith(('http://', 'https://')):
            raise ValueError('only http and https URLs are allowed')
        with requests.get(url, stream=True, timeout=20, headers={'User-Agent': USER_AGENT}) as response:
            response.raise_for_status()
            body = response.raw.read(self.MAX_BYTES, decode_content=True)
        extractor = TextExtractor()
        extractor.feed(body.decode(response.encoding or 'utf-8', errors='replace'))
        text = extractor.text()
        return text[:self.MAX_CHARS] + ('...' if len(text) > self.MAX_CHARS else '') if text else 'the page has no readable text'
