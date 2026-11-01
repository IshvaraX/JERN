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
User: what is the weather in delhi
JERN: <tool_call>{"name": "weather", "arguments": {"city": "Delhi"}}</tool_call>
Tool: Delhi: 31°C, haze, humidity 40%
JERN: It's 31°C and hazy in Delhi right now.
```

Speaker tags: `User:` → user, `JERN:` (or `Assistant:`) → assistant, `Tool:` → tool result; an optional first line `Context: ...` is appended to the system prompt (Saturday's `ConversationFile`). A system message with the persona and tool list is prepended to every conversation at training time.

Edit the file to teach JERN new facts or behaviours, then run `python main.py finetune` (Colab) or `python main.py finetune --local`.

## Chat model (Saturday, `saturday.models`)

JERN's chat model is provided by the [Saturday](https://github.com/vivek09thakur/SATURDAY) framework and configured in `saturday.toml` (`[model]`). The default base is `microsoft/Phi-3.5-mini-instruct` (3.8B): a strong small instruct model with a 128k context and a chat template that supports `system` and `tool` messages. The `<tool_call>{...}</tool_call>` format JERN parses is taught by the fine-tune. Phi-3 fuses its attention and MLP projections (`qkv_proj`, `gate_up_proj`); `main.py` adds those names to Saturday's LoRA target list so the adapter covers them.

**Loading**: the base model is loaded and a LoRA adapter is looked up in `out-jern/`, then on the Hub at `HF_REPO`. An adapter is reused only if its `adapter_config.json` names the configured base model; otherwise (or with `--train`) a fresh one is trained, saved, pushed to `HF_REPO` when set, and merged into the base weights for inference.

**Fine-tuning**: the conversation file is rendered through the tokenizer's chat template; only assistant tokens contribute to the loss. LoRA (rank 16, alpha 32) on the attention and MLP projections of the language model, fp16 autocast with a `GradScaler` on CUDA, gradient checkpointing, `batch_size=1` × `accumulation=2`, AdamW at `2e-4`, gradient clipping at 1.0, `epochs=3`. The 3.8B model in fp16 takes ~7.6 GB of the T4's 15 GB; with gradient checkpointing and `max_length=1024` training fits.

**Generation**: chat template → `model.generate` with `temperature`, `top_p=0.9`, `repetition_penalty`, `no_repeat_ngram_size`; `temperature=0` switches to greedy.

**Backends**: `backend = "transformers"` (above), `backend = "gguf"` (llama.cpp quantized models on CPU, inference only), and `--model_url` which POSTs messages to a running model server. The last is how the assistant runs locally while the model runs on Colab.

| Base model | Fits |
|---|---|
| Qwen2.5-0.5B-Instruct | laptop CPU (slow), any GPU |
| SmolLM2-1.7B-Instruct, Qwen2.5-1.5B-Instruct | T4 and up |
| Phi-3.5-mini-instruct (JERN default), Qwen2.5-3B-Instruct | T4 with LoRA |
| a GGUF quant of any of them (`backend = "gguf"`) | laptop CPU, no training |

## Char model (`model/jern/network.py`, `train.py`, `sample.py`)

A GPT-2-style decoder-only transformer (`JERN`, `JERNConfig`): token + position embeddings, `n_layer` blocks of causal self-attention and GELU MLP with pre-LayerNorm, tied output head. Uses PyTorch's fused scaled-dot-product attention when available.

- `prepare` (`ChatDatasetPreparer`) builds a character vocabulary from the conversation file and writes `train.bin` (all of it) and `val.bin` (the last 10 %, at least 512 chars) as `uint16`, plus `meta.pkl` with the `stoi`/`itos` tables.
- `train` (`Trainer`) is a compact nanoGPT-style loop: random block sampling (`BatchLoader`), cosine LR schedule with warmup (`LearningRateScheduler`), AMP + `GradScaler` for fp16, optional DDP (`DistributedContext`) and `torch.compile`, checkpoints via `Checkpoint` (`ckpt.pt` with model args, optimizer state, iteration and best val loss).
- `sample` / `chat` (`Sampler`, `Chat`) load the checkpoint and tokenizer and generate with temperature and top-k. `Chat` keeps the dialogue as `User: … / JERN: …` text and stops each reply at the first newline.

With the default 4-layer, 128-dim model (~0.8M parameters) training takes a few minutes on CPU; it memorises the conversation file rather than generalising, which is the point of the experiment.

## Configuration

`serve`, `model` and `finetune` read `saturday.toml` through Saturday's `Config` (any key is also a `--key=value` flag; JERN's own settings live in its `[app]` table). The char-model commands keep their dataclasses (`TrainConfig`, `SampleConfig`) in `model/jern/config.py`: `configure(Class, argv)` applies `--key=value` overrides and config files, `seed_everything(seed)` seeds torch and enables TF32, and `RuntimeConfig` adds `device`, `dtype`, `compile` and an `autocast()` context.
