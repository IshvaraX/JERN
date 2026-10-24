# Running on Google Colab

Two ways to use a Colab GPU:

1. **The notebook** `notebook/Just-An-Experimental-Response-Network.ipynb` — open it in Colab, set `REPO_URL` (or upload a zip when prompted) and run the cells. Good for exploring.
2. **The `colab` command** — drives the official [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) from your terminal, so nothing has to be clicked. This page is about that.

## Setup

```powershell
pip install google-colab-cli
colab sessions          # first call opens the Google sign-in flow
```

For `colab -- serve` you also need an OpenSSH client (`ssh`, built into Windows 10+/macOS/Linux). JERN creates an `ed25519` key in `~/.ssh` if none exists — Colab's SSH endpoint accepts only ed25519/ecdsa keys.

## Batch jobs: `train`, `finetune`

```powershell
python main.py colab --gpu=T4 -- train --max_iters=5000 --lr_decay_iters=5000
python main.py colab --gpu=T4 -- finetune --epochs=3
```

What happens (`model/jern/colab.py`, `ColabRunner.run`):

1. The repo is zipped to `export/jern.zip` (excluding `venv`, `.git`, `out-*`, `export`, data binaries, …).
2. Every other Colab session is stopped (`--stop_others`, on by default — the free tier allows one), then `colab new -s jern --gpu T4` provisions the VM; `colab upload` sends the zip.
3. `colab exec` runs a small Python script on the VM: remove leftovers from earlier runs, unzip, `pip install -r requirements.txt`, run `main.py <command>` streaming its output, then zip the output directory. The script prints a sentinel on success; without it the run is treated as failed (IPython swallows exit codes).
4. `colab download` fetches it into `export/<out_dir>/` (`export/out-jern-char/` for `train`, `export/out-jern/` for `finetune`).
5. `colab stop` releases the VM (skip with `--keep`).

Use the result locally: `python main.py serve --out_dir=export/out-jern` or `python main.py sample --out_dir=export/out-jern-char`.

Rough timings on a T4: provisioning ~1 min, install ~1 min, `train` with defaults 1–2 min, `finetune` of the 1.5B model ~10–15 min (the first run also downloads ~3 GB of weights).

## Interactive: `serve`

```powershell
python main.py colab --gpu=T4 -- serve
```

The model runs on the GPU; **the assistant, every tool and the UI run on your machine**. Tools therefore see your files, your shell and your OS, not the VM's.

```
 your machine                                   Colab VM
 ┌──────────────────────────────┐              ┌──────────────────────────┐
 │ browser → ChatServer :8000   │              │ ModelServer :8001        │
 │           Assistant + tools  │  ssh -L 8001 │   JERNChat (Qwen, GPU)   │
 │           RemoteChat ────────┼──────────────┼─→ POST /generate         │
 └──────────────────────────────┘              └──────────────────────────┘
```

Steps (`ColabRunner.serve`):

1. Zip, provision, upload as above.
2. On the VM: `main.py model --model_port=8001 --platform="<your OS>"` is started detached; its log is streamed until `/health` answers (this includes the fine-tune on the first run). `--platform` makes the tool prompt baked into fine-tuning describe *your* OS, not Linux.
3. Locally: `ssh -N -L 8001:127.0.0.1:8001` with `colab ssh --proxy-mode` as the `ProxyCommand` forwards the port.
4. Locally: `main.py serve --model_url=http://127.0.0.1:8001 --web` starts the assistant and opens the browser.
5. On Ctrl+C: the tunnel is closed and the VM is stopped. The fine-tuned model is **not** downloaded by default (it is ~3 GB); pass `--download` to fetch it into `export/out-jern/`, or `--keep` to leave the VM running and `--reuse` next time to skip the fine-tune.

`colab -- model` does steps 1–3 only and leaves the tunnel open, for running `serve --model_url=...` yourself (e.g. with different flags, or several times).

Extra flags after `--` are passed to both sides where they make sense: `-- serve --epochs=3 --temperature=0.5`.

## Behind a corporate proxy

The `colab` CLI talks to `*.prod.colab.dev` through the Python `requests` library, which trusts only its bundled CA list. If your network re-signs TLS (the symptom is `CERTIFICATE_VERIFY_FAILED` on `colab upload`), JERN exports the Windows certificate store plus `certifi` to `export/ca-bundle.pem` and sets `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` for every `colab` call. Set `REQUESTS_CA_BUNDLE` yourself to skip this. The servers themselves use `truststore` for the same reason.

## Limits

- Colab sessions end on idle and usage limits; `serve` on Colab is for trying the model, not for hosting.
- The VM has no public inbound address; everything goes through the authenticated `colab` tunnel.
- `--gpu` availability depends on the Colab plan; use `--gpu=''` for a CPU runtime.
