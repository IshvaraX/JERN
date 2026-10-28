"""Browser automation (moved here from Saturday): open sites, play YouTube, show places on Google Maps."""
import re
import webbrowser
from urllib.parse import quote_plus

import requests

from app.tools.base import Plugin
from constants import GOOGLE_MAPS_URL, SITES, USER_AGENT, YOUTUBE_SEARCH_URL, YOUTUBE_VIDEO_URL


def open_url(url):
    webbrowser.open_new_tab(url)
    return f"opened {url}"


class OpenWebsite(Plugin):

    name = 'open_website'
    description = f"Open a website in the browser. Known names: {', '.join(SITES)}; or pass a full URL or domain."
    parameters = {'site': 'site name, domain or URL'}
    required = ('site',)

    def summary(self, arguments):
        return f"open {arguments.get('site', '')} in the browser"

    def run(self, site):
        key = re.sub(r'[^a-z]', '', site.lower())
        url = SITES.get(key) or (site if site.startswith('http') else f'https://{site}')
        return open_url(url)


class PlayYoutube(Plugin):

    name = 'play_youtube'
    description = 'Play the first YouTube video matching a song or video name.'
    parameters = {'query': 'song or video name'}
    required = ('query',)

    def summary(self, arguments):
        return f"play {arguments.get('query', '')} on YouTube"

    def run(self, query):
        search = YOUTUBE_SEARCH_URL.format(query=quote_plus(query))
        try:
            page = requests.get(search, headers={'User-Agent': USER_AGENT}, timeout=10).text
            match = re.search(r'"videoId":"([\w-]{11})"', page)
        except requests.RequestException:
            match = None
        if match is None:
            open_url(search)
            return f"opened YouTube search results for {query}"
        open_url(YOUTUBE_VIDEO_URL.format(video_id=match.group(1)))
        return f"playing {query} on YouTube"


class SearchLocation(Plugin):

    name = 'search_location'
    description = 'Show a place or address on Google Maps in the browser.'
    parameters = {'place': 'place or address'}
    required = ('place',)

    def summary(self, arguments):
        return f"show {arguments.get('place', '')} on Google Maps"

    def run(self, place):
        return open_url(GOOGLE_MAPS_URL.format(query=quote_plus(place)))
