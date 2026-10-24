import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import requests

from app.tools.base import Plugin
from constants import USER_AGENT


class ResultParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.results = []
        self.field = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get('class', '').split()
        if tag == 'a' and 'result__a' in classes:
            self.results.append({'title': '', 'url': self.target(attrs.get('href', '')), 'snippet': ''})
            self.field = 'title'
        elif tag == 'a' and 'result__snippet' in classes and self.results:
            self.field = 'snippet'

    def handle_endtag(self, tag):
        if tag == 'a':
            self.field = None

    def handle_data(self, data):
        if self.field and self.results:
            self.results[-1][self.field] += data

    @staticmethod
    def target(href):
        # DuckDuckGo wraps links as //duckduckgo.com/l/?uddg=<encoded url>
        query = parse_qs(urlparse(href).query)
        return unquote(query['uddg'][0]) if 'uddg' in query else href


class SearchWeb(Plugin):

    name = 'search_web'
    description = ('Search the web and return the top results with snippets. '
                   'Use it to look up facts or to find the right shell command before running one.')
    parameters = {'query': 'what to search for'}
    required = ('query',)
    URL = 'https://html.duckduckgo.com/html/'
    MAX_RESULTS = 5
    MAX_SNIPPET = 300
    expand = True

    def summary(self, arguments):
        return f"search the web for: {arguments['query']}"

    def run(self, query):
        response = requests.get(self.URL, params={'q': query}, timeout=20, headers={'User-Agent': USER_AGENT})
        response.raise_for_status()
        parser = ResultParser()
        parser.feed(response.text)
        results = [r for r in parser.results if r['title'].strip()][:self.MAX_RESULTS]
        if not results:
            return 'no results found'
        lines = []
        for i, r in enumerate(results, 1):
            snippet = re.sub(r'\s+', ' ', r['snippet']).strip()[:self.MAX_SNIPPET]
            title = r['title'].strip().replace('[', '(').replace(']', ')')
            lines.append(f"{i}. [{title}]({r['url']})\n   {snippet}")
        return '\n'.join(lines)
