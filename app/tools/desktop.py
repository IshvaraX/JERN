"""Desktop control (moved here from Saturday): close the browser tab, shut the computer down."""
import os
import subprocess

from app.tools.base import Plugin


class CloseTab(Plugin):

    name = 'close_tab'
    description = 'Close the current browser tab.'

    def summary(self, arguments):
        return 'close the current browser tab'

    def run(self):
        import pyautogui
        pyautogui.hotkey('ctrl', 'w')
        return 'closed the current tab'


class Shutdown(Plugin):

    name = 'shutdown'
    description = 'Shut down this computer.'
    confirm = True

    def summary(self, arguments):
        return 'shut down the computer'

    def run(self):
        command = ['shutdown', '/s', '/t', '10'] if os.name == 'nt' else ['shutdown', '-h', 'now']
        subprocess.run(command, check=True)
        return 'shutting down in 10 seconds'
