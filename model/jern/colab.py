import os
import platform
import re
import shlex
import shutil
import ssl
import subprocess
import sys
import time
from dataclasses import dataclass

from constants import (ARCHIVE_EXCLUDE, CA_BUNDLE_FILE, CHAR_OUT_DIR, CHAT_OUT_DIR, EXPORT_DIR, HOST, LOCAL_ZIP,
                       MODEL_PORT, NAME, REMOTE_LOG, REMOTE_OK, REMOTE_OUT_ZIP, REMOTE_ROOT, REMOTE_STARTED,
                       REMOTE_TIMEOUT, REMOTE_ZIP, RETRY_DELAY, SERVER_AUTH_OID, SESSION_LINE, SSH_HOST, SSH_KEYS,
                       SSH_OPTIONS, TRANSIENT_ERRORS, UPLOAD_RETRIES)
from model.jern.util import unzip, wait_for_health, zip_dir

REMOTE_SETUP = '''
import os, shutil, subprocess, sys, time, zipfile
print({started!r}, flush=True)
root = {root!r}
shutil.rmtree(root, ignore_errors=True)
for stale in ({out_zip!r}, {log!r}):
    if os.path.exists(stale):
        os.remove(stale)
zipfile.ZipFile({zip!r}).extractall(root)
os.remove({zip!r})
os.chdir(root)
pip = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], capture_output=True, text=True)
if pip.returncode != 0:
    print(pip.stdout[-2000:], pip.stderr[-2000:], flush=True)
    raise SystemExit('pip install failed')
'''

REMOTE_ARCHIVE = '''
import os, shutil, zipfile
print({started!r}, flush=True)
root = {root!r}
if os.path.isdir(os.path.join(root, {out_dir!r})):
    # Model weights do not compress; store them as-is so the archive takes seconds, not minutes.
    with zipfile.ZipFile({out_zip!r}, 'w', zipfile.ZIP_STORED) as zf:
        for dirpath, _, filenames in os.walk(os.path.join(root, {out_dir!r})):
            for filename in filenames:
                full = os.path.join(dirpath, filename)
                zf.write(full, os.path.relpath(full, root))
    print('archived', {out_dir!r}, f'({{os.path.getsize({out_zip!r}) / 1e9:.1f}} GB)')
else:
    print('nothing to download:', {out_dir!r}, 'was not created')
print({ok!r})
'''

REMOTE_RUN = REMOTE_SETUP + '''
proc = subprocess.Popen([sys.executable, '-u', 'main.py', *{args!r}], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end='', flush=True)
if proc.wait() != 0:
    raise SystemExit(f'main.py exited with {{proc.returncode}}')
''' + REMOTE_ARCHIVE

# The model server outlives this cell: detached from the kernel and polled until /health answers.
REMOTE_MODEL = REMOTE_SETUP + '''
import urllib.request
proc = subprocess.Popen([sys.executable, '-u', 'main.py', *{args!r}], stdout=open({log!r}, 'w'),
                        stderr=subprocess.STDOUT, start_new_session=True)
with open({log!r}) as tail:
    while True:
        print(tail.read(), end='', flush=True)
        if proc.poll() is not None:
            raise SystemExit(f'model server exited with {{proc.returncode}}')
        try:
            urllib.request.urlopen('http://127.0.0.1:{port}/health', timeout=2)
            break
        except OSError:
            time.sleep(2)
    print(tail.read(), end='', flush=True)
print('model server ready on port {port}')
print({ok!r})
'''


def render(template, **values):
    return template.format(root=REMOTE_ROOT, zip=REMOTE_ZIP, out_zip=REMOTE_OUT_ZIP,
                           out_zip_base=REMOTE_OUT_ZIP[:-len('.zip')], log=REMOTE_LOG, ok=REMOTE_OK,
                           started=REMOTE_STARTED, **values)


class RemoteError(RuntimeError):
    pass


@dataclass
class ColabConfig:
    session: str = NAME.lower()
    gpu: str = 'T4'  # T4, L4, G4, A100, H100 or '' for a CPU runtime
    tpu: str = ''
    high_mem: bool = False
    timeout: int = REMOTE_TIMEOUT  # seconds the remote command may run
    out: str = ''  # remote directory to download; inferred from the command when empty
    model_port: int = MODEL_PORT  # port the remote model server is forwarded to
    reuse: bool = False  # attach to an existing session instead of creating one
    keep: bool = False  # leave the VM running afterwards
    download: bool = False  # serve: fetch the fine-tuned model (GBs) on exit; train/finetune always download
    stop_others: bool = True  # stop every other session first; the free tier allows only one
    browser: bool = True  # serve: open the local UI in the browser


class ColabRunner:

    DEFAULT_OUT = {'train': CHAR_OUT_DIR, 'finetune': CHAT_OUT_DIR, 'model': CHAT_OUT_DIR, 'serve': CHAT_OUT_DIR}
    USE_WITH = {CHAR_OUT_DIR: 'sample', CHAT_OUT_DIR: 'serve'}
    REMOTE = ('model', 'serve')  # commands whose model process stays up on the VM

    def __init__(self, config, root):
        self.config = config
        self.root = root
        self.export = os.path.join(root, EXPORT_DIR)
        self.colab_path = shutil.which('colab')
        if self.colab_path is None:
            raise SystemExit("the 'colab' CLI is not installed: pip install google-colab-cli")
        os.makedirs(self.export, exist_ok=True)
        # Piped stdout defaults to cp1252 on Windows, which cannot encode progress-bar glyphs from the VM.
        self.env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
        bundle = self.ca_bundle()
        if bundle:
            self.env.update(REQUESTS_CA_BUNDLE=bundle, SSL_CERT_FILE=bundle, CURL_CA_BUNDLE=bundle)

    # --- local environment ---

    def ca_bundle(self):
        # The CLI only trusts certifi; behind a TLS-inspecting proxy the extra CA lives in the OS store (Windows).
        if os.environ.get('REQUESTS_CA_BUNDLE') or not hasattr(ssl, 'enum_certificates'):
            return None
        pems = []
        try:
            import certifi
            with open(certifi.where(), encoding='ascii') as f:
                pems.append(f.read())
        except ImportError:
            pass
        for store in ('ROOT', 'CA'):
            for der, encoding, trust in ssl.enum_certificates(store):
                if encoding == 'x509_asn' and (trust is True or (isinstance(trust, set) and SERVER_AUTH_OID in trust)):
                    pems.append(ssl.DER_cert_to_PEM_cert(der))
        path = os.path.join(self.export, CA_BUNDLE_FILE)
        with open(path, 'w', encoding='ascii') as f:
            f.write('\n'.join(pems))
        return path

    def ssh_key(self):
        ssh_dir = os.path.expanduser('~/.ssh')
        if any(os.path.exists(os.path.join(ssh_dir, name)) for name in SSH_KEYS):
            return
        os.makedirs(ssh_dir, exist_ok=True)
        key = os.path.join(ssh_dir, SSH_KEYS[0])
        print(f"creating SSH key {key} for the Colab tunnel")
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', key], check=True)

    # --- colab CLI ---

    def colab(self, *args, **kwargs):
        command = [self.colab_path, *args]
        print('$', ' '.join(command), flush=True)
        return subprocess.run(command, check=True, env=self.env, **kwargs)

    def session(self, *args, **kwargs):
        return self.colab(*args[:1], '-s', self.config.session, *args[1:], **kwargs)

    def sessions(self):
        output = subprocess.run([self.colab_path, 'sessions'], capture_output=True, text=True, env=self.env).stdout
        return re.findall(SESSION_LINE, output, re.MULTILINE)

    def stop_others(self):
        for name in self.sessions():
            if self.config.reuse and name == self.config.session:
                continue
            print(f"stopping leftover session '{name}'")
            subprocess.run([self.colab_path, 'stop', '-s', name], env=self.env)

    def create(self):
        cfg = self.config
        if cfg.stop_others:
            self.stop_others()
        args = ['new', '-s', cfg.session]
        if cfg.gpu:
            args += ['--gpu', cfg.gpu]
        if cfg.tpu:
            args += ['--tpu', cfg.tpu]
        if cfg.high_mem:
            args.append('--high-mem')
        self.colab(*args)

    def deploy(self):
        source = os.path.join(self.export, LOCAL_ZIP)
        count = zip_dir(self.root, source, ARCHIVE_EXCLUDE)
        print(f"packed {count} files ({os.path.getsize(source) / 1e6:.1f} MB)")
        if self.config.reuse:
            if self.config.stop_others:
                self.stop_others()
        else:
            try:
                self.create()
            except subprocess.CalledProcessError:
                raise SystemExit("could not allocate a runtime: Colab refused (usage or capacity limit); "
                                 "wait and retry, or try --gpu='' for a CPU runtime")
        for attempt in range(1, UPLOAD_RETRIES + 1):
            try:
                self.session('upload', source, REMOTE_ZIP)
                break
            except subprocess.CalledProcessError:
                # A fresh runtime's hostname can take a while to resolve; the CLI gives up immediately.
                if attempt == UPLOAD_RETRIES:
                    self.release()  # a session with nothing on it is useless; do not leave it running
                    raise SystemExit('upload failed; the runtime never became reachable')
                print(f"runtime not reachable yet, retrying in {RETRY_DELAY}s ({attempt}/{UPLOAD_RETRIES})", flush=True)
                time.sleep(RETRY_DELAY)

    def exec(self, script, retries=3):
        command = [self.colab_path, 'exec', '-s', self.config.session, '--timeout', str(self.config.timeout)]
        for attempt in range(1, retries + 1):
            print('$', ' '.join(command), flush=True)
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                       env=self.env)
            process.stdin.write(script)
            process.stdin.close()
            ok, started, output = False, False, []
            for line in process.stdout:
                if REMOTE_OK in line:
                    ok = True
                elif REMOTE_STARTED in line:
                    started = True
                else:
                    output.append(line)
                    print(line, end='', flush=True)
            if process.wait() == 0 and ok:
                return
            # The kernel's websocket is flaky right after provisioning; only retry if our script never began.
            transient = not started and any(marker in ''.join(output) for marker in TRANSIENT_ERRORS)
            if not transient or attempt == retries:
                raise RemoteError('remote command failed')
            print(f"kernel not ready, retrying in {RETRY_DELAY}s ({attempt}/{retries})", flush=True)
            time.sleep(RETRY_DELAY)

    def download(self, out_dir):
        result = os.path.join(self.export, os.path.basename(REMOTE_OUT_ZIP))
        try:
            self.session('download', REMOTE_OUT_ZIP, result)
        except subprocess.CalledProcessError:
            print(f"no output archive to download for {out_dir}")
            return
        target = os.path.join(self.export, out_dir)
        shutil.rmtree(target, ignore_errors=True)
        unzip(result, self.export)
        print(f"downloaded to {target}")
        if out_dir in self.USE_WITH:
            print(f"use it with: python main.py {self.USE_WITH[out_dir]} --out_dir={EXPORT_DIR}/{out_dir}")

    def release(self):
        if self.config.keep:
            print(f"session '{self.config.session}' kept running; stop it with: colab stop -s {self.config.session}")
        else:
            self.session('stop')

    # --- commands ---

    @staticmethod
    def option(args, key, default):
        for arg in args:
            if arg.startswith(f'--{key}='):
                return arg.split('=', 1)[1].strip('\'"')
        return default

    def out_dir(self, args):
        return self.config.out or self.option(args, 'out_dir', self.DEFAULT_OUT.get(args[0], CHAT_OUT_DIR))

    def run(self, args):
        if args[0] in self.REMOTE:
            return self.serve(args)
        out_dir = self.out_dir(args)
        self.deploy()
        try:
            self.exec(render(REMOTE_RUN, args=args, out_dir=out_dir))
            self.download(out_dir)
        finally:
            self.release()

    def tunnel(self, port):
        proxy = [self.colab_path, 'ssh', '--proxy-mode', '-s', self.config.session]
        proxy = subprocess.list2cmdline(proxy) if os.name == 'nt' else shlex.join(proxy)
        options = [f'ProxyCommand={proxy}', f'UserKnownHostsFile={os.devnull}', *SSH_OPTIONS]
        command = ['ssh', '-N', *(flag for option in options for flag in ('-o', option)),
                   '-L', f'{port}:127.0.0.1:{port}', SSH_HOST]
        print('$', ' '.join(command), flush=True)
        process = subprocess.Popen(command, env=self.env)
        if not wait_for_health(f'http://{HOST}:{port}', process, attempts=60):
            process.terminate()
            raise SystemExit('ssh tunnel did not come up')
        return process

    def serve(self, args):
        # Model on the GPU VM; assistant, tools and web UI on this machine, talking over an SSH tunnel.
        if shutil.which('ssh') is None:
            raise SystemExit("'ssh' is not installed (on Windows enable the OpenSSH Client optional feature)")
        port = self.config.model_port
        out_dir = self.out_dir(args)
        passthrough = [a for a in args[1:] if not a.startswith(('--port=', '--web', '--model_url=', '--model_port='))]
        remote = ['model', f'--model_port={port}', f'--platform={platform.system()} {platform.release()}', *passthrough]
        self.ssh_key()
        self.deploy()
        tunnel = None
        try:
            self.exec(render(REMOTE_MODEL, args=remote, port=port))
            tunnel = self.tunnel(port)
            model_url = f'http://{HOST}:{port}'
            if args[0] == 'model':
                print(f"model server forwarded to {model_url}   (Ctrl+C to stop)\n"
                      f"in another terminal: python main.py serve --model_url={model_url}", flush=True)
                tunnel.wait()
            else:
                local = [sys.executable, 'main.py', 'serve', f'--model_url={model_url}', *args[1:]]
                if self.config.browser and '--web' not in local:
                    local.append('--web')
                print('$', ' '.join(local), flush=True)
                subprocess.run(local, cwd=self.root)
        except KeyboardInterrupt:
            print('\nstopping')
        finally:
            if tunnel is not None and tunnel.poll() is None:
                tunnel.terminate()
            if not self.config.keep and self.config.download:
                try:
                    print(f"downloading {out_dir} (several GB; Ctrl+C again to skip)", flush=True)
                    self.exec(render(REMOTE_ARCHIVE, out_dir=out_dir))
                    self.download(out_dir)
                except (RemoteError, subprocess.CalledProcessError, KeyboardInterrupt):
                    print(f"skipped downloading {out_dir}")
            elif not self.config.keep:
                print(f"not downloading {out_dir} (pass --download to keep the fine-tuned model, "
                      f"or --keep to leave the VM running and --reuse next time)")
            try:
                self.release()
            except KeyboardInterrupt:
                self.release()
