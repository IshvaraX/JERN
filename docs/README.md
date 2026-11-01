# JERN — Just An Experimental Response Network

JERN is a small personal AI assistant by Vivek Thakur. It has two models and one app:

| Part | What it is | Where |
|---|---|---|
| **Chat model** | `Qwen/Qwen2.5-1.5B-Instruct` fine-tuned on `data/chat/conversations.txt` so it answers as JERN and knows how to call tools | `model/jern/chat.py` |
| **Char model** | A from-scratch GPT-style transformer trained character-by-character on the same file (an experiment, not the assistant) | `model/jern/network.py`, `train.py`, `sample.py` |
| **App** | FastAPI server + web UI + tool plugins (web search, page reading, downloads, browser, weather, email) + any MCP servers you configure (Figma, GitHub, …) | `app/server/`, `app/tools/`, `app/ui/` |

The model can run on this machine or on a Google Colab GPU; the tools **always run on this machine**, so JERN can read your files and run your commands even when the model is in the cloud.

## Quick start

```powershell
python -m venv venv; .\venv\Scripts\activate
pip install -r requirements.txt

python main.py                          # web UI at http://127.0.0.1:8000 (fine-tunes on first run)
python main.py colab --gpu=T4 -- serve  # same UI, but the model runs on a Colab T4
```

The first run downloads the base model and fine-tunes it (minutes on a GPU, an hour or more on CPU). The result is saved in `out-jern/` and reused afterwards.

## Documents

- [cli.md](cli.md) — every `main.py` command and flag
- [model.md](model.md) — the two models, the data format, training and fine-tuning
- [server.md](server.md) — HTTP API, sessions, the tool-call protocol, writing a plugin
- [colab.md](colab.md) — running on Google Colab, the SSH tunnel, TLS behind a proxy
