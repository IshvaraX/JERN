# Running on Google Colab

Two ways to use a Colab GPU:

1. **The notebook** `notebook/Just-An-Experimental-Response-Network.ipynb` — open it in Colab, set `REPO_URL` (or upload a zip when prompted) and run the cells. Good for exploring.
2. **Saturday's Colab runner** — `serve` and `finetune` go to Colab by default, driving the official [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) from your terminal so nothing has to be clicked. This page is about that.

## Setup

```powershell
colab sessions          # installed with saturday-ai; the first call opens the Google sign-in flow
```

For `serve` you also need an OpenSSH client (`ssh`, built into Windows 10+/macOS/Linux). Saturday creates an `ed25519` key in `~/.ssh` if none exists — Colab's SSH endpoint accepts only ed25519/ecdsa keys. Set `HF_REPO` and `HF_TOKEN` in `.env` so the trained adapter is pushed to the Hub and later runs skip training.

## Batch jobs: `finetune`, `train --colab`

```powershell
python main.py finetune --epochs=3
python main.py train --colab --max_iters=5000 --lr_decay_iters=5000
```

What happens (`saturday.colab.ColabRunner`):

1. The repo is zipped to `export/saturday-upload.zip` under `project/`, with the installed `saturday` package alongside under `saturday/` so the VM runs exactly the framework version you have. `venv`, `.git`, `out-*`, `export`, `workspace`, data binaries and the `[colab] exclude` list from `saturday.toml` (`mcp.json`, `mcp-servers`, `notebook`, `docs`) stay home.
2. Every other Colab session is stopped (the free tier allows one), then `colab new -s jern --gpu T4` provisions the VM; `colab upload` sends the zip.
3. `colab exec` runs a small Python script on the VM: unzip, `pip install transformers peft huggingface_hub`, run the command (`python -m saturday finetune --local --platform="<your OS>"`, or `main.py train` for the char model) streaming its output. The script prints a sentinel on success; without it the run is treated as failed (IPython swallows exit codes).
4. `finetune` pushes the adapter to `HF_REPO` when set, otherwise `colab download` fetches `out-jern/`; `train --colab` always downloads `out-jern-char/`.
5. `colab stop` releases the VM (skip with `--keep`).

Rough timings on a T4: provisioning ~1 min, install ~1 min, `train` with defaults 1–2 min, LoRA `finetune` of the 1.5B model ~5 min (the first run also downloads ~3 GB of weights).

## Interactive: `serve`

```powershell
python main.py                 # same as: python main.py serve
```

The model runs on the GPU; **the assistant, every tool and the UI run on your machine**. Tools therefore see your files, your shell and your OS, not the VM's.

```
 your machine                                   Colab VM
 ┌──────────────────────────────┐              ┌──────────────────────────┐
 │ browser → ChatServer :8000   │              │ saturday model :8001     │
 │           Assistant + tools  │  ssh -L 8001 │   Qwen + LoRA (GPU)      │
 │           RemoteChat ────────┼──────────────┼─→ POST /generate         │
 └──────────────────────────────┘              └──────────────────────────┘
```

Steps (`ColabRunner.serve`):

1. Zip, provision, upload as above.
2. On the VM: `python -m saturday model --local --platform="<your OS>"` is started detached; its log is streamed until `/health` answers (this includes the fine-tune on the first run). `--platform` makes the tool prompt baked into fine-tuning describe *your* OS, not Linux.
3. Locally: `ssh -N -L 8001:127.0.0.1:8001` with `colab ssh --proxy-mode` as the `ProxyCommand` forwards the port.
4. Locally: `main.py serve --model_url=http://127.0.0.1:8001` starts the assistant (add `--web` to open the browser).
5. On Ctrl+C: the tunnel is closed and the VM is stopped. Without `HF_REPO` the adapter is lost with the VM; set it, or `--keep` the VM and `--reuse` it next time.

Flags are passed to both sides where they make sense: `python main.py --epochs=3 --temperature=0.5`.

## Behind a corporate proxy

The `colab` CLI talks to `*.prod.colab.dev` through the Python `requests` library, which trusts only its bundled CA list. If your network re-signs TLS (the symptom is `CERTIFICATE_VERIFY_FAILED` on `colab upload`), Saturday exports the Windows certificate store plus `certifi` to `export/ca-bundle.pem` and sets `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` for every `colab` call. Set `REQUESTS_CA_BUNDLE` yourself to skip this. The servers themselves use `truststore` for the same reason.

## Limits

- Colab sessions end on idle and usage limits; `serve` on Colab is for trying the model, not for hosting.
- The VM has no public inbound address; everything goes through the authenticated `colab` tunnel.
- `--gpu` availability depends on the Colab plan; use `--gpu=''` for a CPU runtime.
