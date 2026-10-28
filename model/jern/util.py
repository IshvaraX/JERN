import fnmatch
import os
import socket
import time
import zipfile


def wait_for_port(host, port, process=None, attempts=120, interval=1.0):
    for _ in range(attempts):
        if process is not None and process.poll() is not None:
            raise SystemExit(f"{process.args[0]} exited with {process.returncode}")
        try:
            socket.create_connection((host, port), timeout=1).close()
            return True
        except OSError:
            time.sleep(interval)
    return False


def wait_for_health(url, process=None, attempts=120, interval=1.0):
    import requests
    for _ in range(attempts):
        if process is not None and process.poll() is not None:
            raise SystemExit(f"{process.args[0]} exited with {process.returncode}")
        try:
            if requests.get(f'{url}/health', timeout=2).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)
    return False


def zip_dir(root, path, exclude=()):
    def excluded(name):
        return any(fnmatch.fnmatch(name, pattern) for pattern in exclude)

    count = 0
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not excluded(d)]
            for filename in filenames:
                if not excluded(filename):
                    full = os.path.join(dirpath, filename)
                    zf.write(full, os.path.relpath(full, root))
                    count += 1
    return count


def unzip(path, target):
    zipfile.ZipFile(path).extractall(target)
