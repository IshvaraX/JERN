# Command line

Everything goes through `main.py`:

```
python main.py [command] [flags...]
```

With no command, `serve` is assumed. `python main.py --help` prints a summary.

## Flags

`serve`, `model` and `finetune` are configured by `saturday.toml` (Saturday's project file); every key in it can be
overridden on the command line. The char-model commands (`prepare`, `train`, `sample`, `chat`) keep their own config
dataclasses.

| Form | Meaning |
|---|---|
| `--key=value` | set a field; the value is parsed as a Python literal (`--epochs=3`, `--temperature=0.7`, `--gpu=''`) or kept as a string if it is not one |
| `--flag` | set a boolean field to `True` (`--web`, `--train`, `--local`) |
| `path.py` | char-model commands only: a Python file whose top-level variables override fields of the same name |

Unknown keys and type mismatches are errors.

## Commands

### `serve` (default)

Runs the assistant: web UI, HTTP API and tool plugins on this machine. Where the model runs depends on flags:

| | |
|---|---|
| `python main.py` | model on a Colab GPU: Saturday provisions a T4, starts the model server there, opens an SSH tunnel and runs `serve --model_url=...` here |
| `python main.py serve --local` | model on this machine (trains the LoRA adapter on the first run) |
| `python main.py serve --model_url=URL` | generation delegated to a model server you started yourself |

| Flag | Default (saturday.toml) | Purpose |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `8000` | where the UI/API listen (`[app]`) |
| `--web` | off | open the browser once the server is up |
| `--local` | off | run the model here instead of on Colab |
| `--model_url` | `''` | use a running model server; no model is loaded locally |
| `--model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | base model (any HF chat model) |
| `--backend` | `transformers` | or `gguf` for llama.cpp quantized models (CPU, inference only, always local) |
| `--out_dir` | `out-jern` | where the LoRA adapter is saved / loaded from; `HF_REPO` in `.env` keeps it on the Hub |
| `--train` | off | train the adapter again even if one exists |
| `--lora_rank`, `--lora_alpha`, `--lora_dropout` | `16`, `32`, `0.05` | adapter size |
| `--epochs`, `--learning_rate`, `--batch_size`, `--accumulation`, `--max_length` | `3`, `2e-4`, `1`, `2`, `1024` | fine-tuning; an optimizer step happens every `accumulation` batches |
| `--temperature`, `--repetition_penalty`, `--max_new_tokens` | `0.3`, `1.15`, `300` | generation |
| `--max_history` | `5` | user turns kept in context |
| `--max_tool_calls`, `--max_tool_result` | `3`, `1500` | tool loop limits per turn |
| `--workspace`, `--downloads` | `workspace`, `workspace/downloads` | sandbox for file tools (`[app]`) |
| `--command_timeout` | `60` | seconds a shell command may run (`[app]`) |
| `--mcp` | `mcp.json` | MCP servers to expose as tools (see server.md); `''` disables (`[app]`) |
| `--gpu`, `--session`, `--keep`, `--reuse` | `T4`, `jern`, off, off | Colab runtime (see colab.md) |
| `--platform` | this machine | OS name baked into the tool prompt (Saturday sets it so a Linux VM fine-tunes for a Windows user) |
| `--persona` | `[assistant] persona` | persona text |
| `--device`, `--seed` | auto, `1337` | |

### `model`

Runs only the model as an HTTP service (`POST /generate`, `GET /health`) on `--model_port` (default `8001`); this
is what Saturday starts on the Colab VM. Needs `--local`. Pair it with `serve --model_url=http://host:8001`.

### `finetune`

Trains the LoRA adapter and exits: on Colab by default (pushed to `HF_REPO` or downloaded to `out-jern/`), on this
machine with `--local`. Same flags as `serve`; `--train` is implied.

### `prepare`

Encodes `data/chat/conversations.txt` into `train.bin`, `val.bin` and `meta.pkl` for the char-level model. `train` runs this automatically if the files are missing.

### `train`

Trains the from-scratch char-level transformer. Config: `TrainConfig`. Add `--colab` to run it on a Colab GPU and
download `out-jern-char/` afterwards.

| Flag | Default |
|---|---|
| `--out_dir` | `out-jern-char` |
| `--max_iters`, `--lr_decay_iters` | `2000`, `2000` |
| `--n_layer`, `--n_head`, `--n_embd`, `--block_size` | `4`, `4`, `128`, `128` |
| `--batch_size`, `--gradient_accumulation_steps` | `16`, `1` |
| `--learning_rate`, `--min_lr`, `--warmup_iters`, `--weight_decay`, `--dropout` | `1e-3`, `1e-4`, `100`, `0.1`, `0.0` |
| `--eval_interval`, `--eval_iters`, `--log_interval` | `250`, `20`, `50` |
| `--init_from` | `scratch` (or `resume`) |
| `--device`, `--dtype`, `--compile` | auto, auto, off |
| `--wandb_log`, `--wandb_project`, `--wandb_run_name` | off, `jern`, `chat` |

### `sample` / `chat`

`sample` prints `--num_samples` continuations of `--start` from the char model in `--out_dir`; `chat` is a terminal conversation with it (`User:` / `JERN:` turns). Config: `SampleConfig` — `--temperature`, `--top_k`, `--max_new_tokens`, `--seed`.

### Colab

There is no separate `colab` command any more: `serve` and `finetune` go to Colab unless `--local` or `--model_url`
is given, and `train --colab` runs the char model there. Runtime flags live in `[colab]` of `saturday.toml`:

| Flag | Default | Purpose |
|---|---|---|
| `--gpu` | `T4` | `T4`, `L4`, `A100`, or `''` for CPU |
| `--session` | `jern` | Colab session name |
| `--reuse` | off | attach to an existing session instead of creating one |
| `--keep` | off | leave the VM running afterwards |
| `--remote_timeout` | `10800` | seconds the remote command may run |
| `--model_port` | `8001` | local port the remote model server is forwarded to |

For `serve` the model stays up on the VM, an SSH tunnel forwards it to `127.0.0.1:8001`, and the assistant with its
tools and UI runs **locally** against it. See [colab.md](colab.md).
