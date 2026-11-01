import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from constants import FULL_NAME, NAME  # noqa: E402

USAGE = f"""{NAME}: {FULL_NAME} (built on Saturday)

  python main.py [serve]                 assistant: web UI + tools here, model on a Colab GPU (default)
  python main.py serve --local           everything on this machine
  python main.py serve --model_url=URL   assistant here, generation delegated to a model server
  python main.py model --local           model server only (POST /generate); fine-tunes on first run
  python main.py finetune                train the LoRA adapter on Colab (--local to do it here)
  python main.py prepare                 encode data/chat/conversations.txt for the char-level model
  python main.py train                   train the from-scratch char-level model (--colab to do it on a GPU)
  python main.py sample | chat           sample from / talk to the char-level model

Flags are --key=value overrides of saturday.toml (serve/model/finetune) or of the command's config dataclass
(prepare/train/sample/chat). Full reference: docs/cli.md
"""

COMMANDS = ('serve', 'model', 'finetune', 'prepare', 'train', 'sample', 'chat')


def saturday_config(argv):
    import truststore

    from saturday.config import configure
    from saturday.utils import load_env

    truststore.inject_into_ssl()
    load_env()
    extend_lora_targets()
    config = configure(argv, ROOT)
    if not config.hf_repo and os.environ.get('HF_TOKEN'):
        config.hf_repo = default_hf_repo(config.model_name)
        if config.hf_repo:
            os.environ['HF_REPO'] = config.hf_repo  # forwarded to the Colab VM so it pushes there too
    return config


def extend_lora_targets():
    """Phi-3 fuses its attention and MLP projections (qkv_proj, gate_up_proj); Saturday only lists the Llama names."""
    from saturday.models import transformers

    for name in ('qkv_proj', 'gate_up_proj'):
        if name not in transformers.LORA_TARGETS:
            transformers.LORA_TARGETS = (*transformers.LORA_TARGETS, name)


def default_hf_repo(model_name):
    """<hub user>/jern-<base model>, so each base model keeps its own adapter and is trained once."""
    from huggingface_hub import HfApi

    try:
        user = HfApi().whoami()['name']
    except Exception as error:
        print(f"HF_REPO not set and the Hub account could not be read ({type(error).__name__}); the adapter stays local")
        return ''
    slug = re.sub(r'[^a-z0-9]+', '-', model_name.rsplit('/', 1)[-1].lower()).strip('-')
    repo = f"{user}/jern-{slug}"
    print(f"adapter repo: {repo} (set HF_REPO in .env to override)")
    return repo


def serve(argv):
    config = saturday_config(argv)
    if config.runs_remote():
        from saturday.colab import ColabRunner

        return ColabRunner(config, ROOT).serve(argv, [sys.executable, 'main.py', 'serve', *argv])
    from app.server.api import ChatServer
    from app.server.assistant import Assistant, EventLog
    from app.tools import ToolRegistry

    tools = ToolRegistry(config)
    print(f"tools: {', '.join(p.name for p in tools)}")
    log = EventLog()
    ChatServer(Assistant(config, tools, log), log).run(config.host, config.port, config.web)


def model(argv):
    from saturday.config import Log
    from saturday.models.server import ModelServer

    config = saturday_config(argv)
    if config.runs_remote():
        raise SystemExit('the model server runs where the model is; add --local (or use `serve` for the Colab flow)')
    ModelServer(config, Log()).serve()


def finetune(argv):
    from saturday.config import Log

    config = saturday_config(argv)
    config.train = True
    if config.runs_remote():
        from saturday.colab import ColabRunner

        return ColabRunner(config, ROOT).run('finetune', argv)
    from saturday.assistant import build_system_prompt
    from saturday.models import chat_backend
    from saturday.plugins import registry

    chat_backend(config, Log()).load(build_system_prompt(config, registry(config)))


def prepare(argv):
    from constants import DATA_DIR, DATASET
    from model.jern.data import ChatDatasetPreparer

    ChatDatasetPreparer(os.path.join(ROOT, DATA_DIR, DATASET)).run()


def train(argv):
    if '--colab' in argv:
        from constants import CHAR_OUT_DIR
        from saturday.colab import ColabRunner

        flags = [a for a in argv if a != '--colab']
        return ColabRunner(saturday_config([]), ROOT).run_remote(['main.py', 'train', *flags], download=CHAR_OUT_DIR)
    from model.jern.config import configure
    from model.jern.train import TrainConfig, Trainer

    config = configure(TrainConfig, argv)
    data_dir = os.path.join(config.data_dir, config.dataset)
    if not os.path.exists(os.path.join(data_dir, 'train.bin')):
        print(f"{data_dir} is not prepared yet, preparing it first")
        prepare([])
    trainer = Trainer(config)
    try:
        trainer.train()
    finally:
        trainer.close()


def sampler(argv):
    from model.jern.config import configure
    from model.jern.sample import SampleConfig, Sampler

    return Sampler(configure(SampleConfig, argv))


def sample(argv):
    sampler(argv).run()


def chat(argv):
    sampler(argv).chat().run()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ('-h', '--help', 'help'):
        print(USAGE)
        return
    command = argv.pop(0) if argv and argv[0] in COMMANDS else 'serve'
    globals()[command](argv)


if __name__ == '__main__':
    main()
