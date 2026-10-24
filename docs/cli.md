# Command line

Everything goes through `main.py`:

```
python main.py [command] [flags...]
```

With no command, `serve` is assumed. `python main.py --help` prints a summary.

## Flags

Every command has a config dataclass. Any field can be overridden on the command line:

| Form | Meaning |
|---|---|
| `--key=value` | set a field; the value is parsed as a Python literal (`--epochs=3`, `--temperature=0.7`, `--gpu=''`) or kept as a string if it is not one |
| `--flag` | set a boolean field to `True` (`--web`, `--train`) |
| `path.py` | a bare argument is a Python file; its top-level variables override fields of the same name |

Unknown keys and type mismatches are errors.

## Commands

### `serve` (default)

Runs the assistant: web UI, HTTP API, tool plugins and — unless `--model_url` is given — the chat model itself. Fine-tunes the base model into `out_dir` on the first run.

Config: `AppConfig` (includes everything in `ChatConfig`).

| Flag | Default | Purpose |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `8000` | where the UI/API listen |
| `--web` | off | open the browser once the server is up |
| `--model_url` | `''` | delegate generation to a `model` server at this URL (e.g. the Colab tunnel); no model is loaded locally |
| `--model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | base model |
| `--out_dir` | `out-jern` | where the fine-tuned model is saved / loaded from |
| `--train` | off | fine-tune again even if `out_dir` exists |
| `--train_layers` | `6` | how many final transformer layers to fine-tune (`0` = all) |
| `--epochs`, `--learning_rate`, `--batch_size`, `--accumulation`, `--max_length` | `2`, `1e-4`, `1`, `2`, `1024` | fine-tuning; an optimizer step happens every `accumulation` batches |
| `--temperature`, `--repetition_penalty`, `--max_new_tokens` | `0.3`, `1.1`, `300` | generation |
| `--max_history` | `5` | user turns kept in context |
| `--max_tool_calls`, `--max_tool_result` | `3`, `1500` | tool loop limits per turn |
| `--workspace`, `--downloads` | `workspace`, `workspace/downloads` | sandbox for file tools |
| `--command_timeout` | `60` | seconds a shell command may run |
| `--mcp` | `mcp.json` | MCP servers to expose as tools (see server.md); `''` disables |
| `--platform` | this machine | OS name baked into the tool prompt (set by `colab -- serve` so a Linux VM fine-tunes for a Windows user) |
| `--system_prompt` | `constants.PERSONA` | persona text |
| `--device`, `--seed` | auto, `1337` | |

### `model`

Runs only the model as an HTTP service (`POST /generate`, `GET /health`) on `--model_port` (default `8001`). Fine-tunes on first run exactly like `serve`. Pair it with `serve --model_url=http://host:8001` on another machine or in another terminal. Takes the same flags as `serve`.

### `finetune`

Fine-tunes the chat model into `out_dir` and exits. Same flags as `serve`; `--train` is implied.

### `prepare`

Encodes `data/chat/conversations.txt` into `train.bin`, `val.bin` and `meta.pkl` for the char-level model. `train` runs this automatically if the files are missing.

### `train`

Trains the from-scratch char-level transformer. Config: `TrainConfig`.

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

### `colab`

Runs another command on a Google Colab GPU. Flags before `--` configure the runtime; everything after `--` is the command to run there:

```
python main.py colab [--gpu=T4] [--session=jern] [--keep] [--reuse] -- train --max_iters=5000
python main.py colab --gpu=L4 -- finetune --epochs=3
python main.py colab --gpu=T4 -- serve
```

| Flag | Default | Purpose |
|---|---|---|
| `--gpu` | `T4` | `T4`, `L4`, `G4`, `A100`, `H100`, or `''` for CPU |
| `--tpu`, `--high_mem` | `''`, off | |
| `--session` | `jern` | Colab session name |
| `--reuse` | off | attach to an existing session instead of creating one |
| `--stop_others` | on | stop every other Colab session first (the free tier allows one GPU session at a time) |
| `--keep` | off | leave the VM running afterwards |
| `--download` | off | `serve`: on exit, fetch the fine-tuned model (~3 GB) into `export/`; `train`/`finetune` always download |
| `--timeout` | `10800` | seconds the remote command may run |
| `--out` | inferred | which remote directory to download into `export/` |
| `--model_port` | `8001` | local port the remote model server is forwarded to |
| `--browser` | on | `serve`: open the UI in the browser |

For `train` / `finetune` the output directory is downloaded to `export/<out_dir>/` and the VM is stopped. For `serve` (and `model`) the model stays up on the VM, an SSH tunnel forwards it to `127.0.0.1:8001`, and the assistant with its tools and UI runs **locally** against it. See [colab.md](colab.md).
