import os
from urllib.parse import urlparse

import requests

from app.tools.base import Plugin, Sandbox


class Download(Plugin):

    name = 'download'
    description = 'Download an http(s) URL into the downloads folder.'
    parameters = {'url': 'http(s) URL of the file', 'filename': 'optional name to save as'}
    required = ('url',)
    confirm = True
    MAX_BYTES = 50 * 1024 * 1024

    def summary(self, arguments):
        return f"download {arguments['url']}"

    def run(self, url, filename=''):
        if not url.startswith(('http://', 'https://')):
            raise ValueError('only http and https URLs are allowed')
        downloads = Sandbox(self.config.downloads)
        name = os.path.basename(filename) or os.path.basename(urlparse(url).path) or 'download'
        target = downloads.resolve(name)
        size = 0
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(target, 'wb') as f:
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > self.MAX_BYTES:
                        f.close()
                        os.remove(target)
                        raise ValueError('file is larger than 50 MB')
                    f.write(chunk)
        return f"saved {name} ({size / 1024:.1f} KB) to {self.workspace.relative(target)}"
