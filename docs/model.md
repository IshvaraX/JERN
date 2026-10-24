# Models and data

## Training data: `data/chat/conversations.txt`

One file, plain text. Conversations are separated by a line containing only `---`. Each message starts with a speaker tag; a message may span several lines (including blank ones) until the next tag.

```
User: hello
JERN: Hello! I'm JERN, a small AI assistant built by Vivek Thakur. ...
User: what can you do
JERN: Quite a few things! Here's a quick overview:
- Answer questions
- Search the web
---
User: list my files
JERN: <tool_call>{"name": "list_files", "arguments": {"path": "."}}</tool_call>
Tool: workspace:
notes/
todo.md (0.1 KB)
JERN: You have a notes folder and a todo.md file.
```

Speaker tags (`constants.SPEAKERS`): `User:` → user, `JERN:` (or legacy `AI:`) → assistant, `Tool:` → tool result. A system message with the persona and tool list is prepended to every conversation at training time (`ConversationFile`, `model/jern/data.py`).

Edit the file to teach JERN new facts or behaviours, then run `python main.py finetune` (or `colab -- finetune`).

## Chat model (`model/jern/chat.py`)

`JERNChat` wraps a Hugging Face causal LM. The default base is `Qwen/Qwen2.5-1.5B-Instruct`, chosen because its chat template natively uses the same `<tool_call>{...}</tool_call>` format JERN parses and understands `tool` role messages.

**Loading** (`load(system_prompt)`): if `out_dir` already holds a fine-tuned model and `--train` is not set, it is loaded; otherwise the base model is loaded and fine-tuned first.

**Fine-tuning** (`finetune`): the conversation file is rendered through the tokenizer's chat template (`ConversationDataset`); only assistant tokens contribute to the loss (`IGNORE_INDEX` elsewhere). Only the last `train_layers` transformer blocks plus the final norm are trained (default 6 of 28). On CUDA the forward/backward runs under fp16 autocast with a `GradScaler` (weights stay fp32), with gradient checkpointing and `batch_size=1` × `accumulation=2`, which keeps the 1.5B model under ~10 GB on a 15 GB T4. `--train_layers=0` trains everything. AdamW, gradient clipping at 1.0, `epochs` passes.

**Generation** (`generate(messages)`): chat template → `model.generate` with `temperature`, `top_p=0.9`, `repetition_penalty`; `temperature=0` switches to greedy.

`RemoteChat` has the same interface but POSTs the messages to a `ModelServer` (`/generate`). `chat_backend(config)` picks one based on `config.model_url`. This is how the assistant runs locally while the model runs on Colab.

Memory guide for fine-tuning (defaults: fp32 weights, fp16 autocast, `train_layers=6`, batch 1 × 1024 tokens, checkpointing):

| Base model | Weights | Fits |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 2 GB | laptop CPU (slow), any GPU |
| Qwen2.5-1.5B-Instruct (default) | 6 GB | T4 and up |
| Qwen2.5-3B-Instruct | 12 GB | L4/A100; on a T4 lower `--train_layers` and `--max_length` |

## Char model (`model/jern/network.py`, `train.py`, `sample.py`)

A GPT-2-style decoder-only transformer (`JERN`, `JERNConfig`): token + position embeddings, `n_layer` blocks of causal self-attention and GELU MLP with pre-LayerNorm, tied output head. Uses PyTorch's fused scaled-dot-product attention when available.

- `prepare` (`ChatDatasetPreparer`) builds a character vocabulary from the conversation file and writes `train.bin` (all of it) and `val.bin` (the last 10 %, at least 512 chars) as `uint16`, plus `meta.pkl` with the `stoi`/`itos` tables.
- `train` (`Trainer`) is a compact nanoGPT-style loop: random block sampling (`BatchLoader`), cosine LR schedule with warmup (`LearningRateScheduler`), AMP + `GradScaler` for fp16, optional DDP (`DistributedContext`) and `torch.compile`, checkpoints via `Checkpoint` (`ckpt.pt` with model args, optimizer state, iteration and best val loss).
- `sample` / `chat` (`Sampler`, `Chat`) load the checkpoint and tokenizer and generate with temperature and top-k. `Chat` keeps the dialogue as `User: … / JERN: …` text and stops each reply at the first newline.

With the default 4-layer, 128-dim model (~0.8M parameters) training takes a few minutes on CPU; it memorises the conversation file rather than generalising, which is the point of the experiment.

## Configuration (`model/jern/config.py`)

Each command has a dataclass (`AppConfig`, `TrainConfig`, `SampleConfig`, `ColabConfig`). `configure(Class, argv)` creates one and applies `--key=value` overrides and config files; `seed_everything(seed)` seeds torch and enables TF32. `RuntimeConfig` adds `device`, `dtype`, `compile` and an `autocast()` context used by the char model.
