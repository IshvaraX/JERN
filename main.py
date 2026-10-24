import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # all default paths (data/, out-*/, workspace/) are relative to the repo root
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # less fragmentation when fine-tuning

from constants import DATA_DIR, DATASET, FULL_NAME, NAME  # noqa: E402

USAGE = f"""{NAME}: {FULL_NAME}

  python main.py [serve]                 assistant: web UI + tools + model, all on this machine
  python main.py serve --model_url=URL   assistant here, generation delegated to a model server
  python main.py model                   model server only (POST /generate); fine-tunes on first run
  python main.py finetune                fine-tune the instruct model into out-jern/ and exit
  python main.py prepare                 encode data/chat/conversations.txt for the char-level model
  python main.py train                   train the from-scratch char-level model
  python main.py sample | chat           sample from / talk to the char-level model
  python main.py colab [opts] -- CMD     run CMD on a Colab GPU; `-- serve` keeps the model there and
                                         runs the assistant, tools and UI locally through an SSH tunnel

Flags are --key=value overrides of the command's config dataclass; a bare path is a Python config file.
Full reference: docs/cli.md
"""

COMMANDS = ('serve', 'model', 'finetune', 'prepare', 'train', 'sample', 'chat', 'colab')


def app_config(argv):
    import truststore

    from app.server.assistant import AppConfig
    from model.jern.config import configure, seed_everything

    truststore.inject_into_ssl()  # trust the OS certificate store (needed behind corporate TLS proxies)
    config = configure(AppConfig, argv)
    seed_everything(config.seed)
    return config


def serve(argv):
    from app.server.api import ChatServer
    from app.server.assistant import Assistant, EventLog
    from app.tools import ToolRegistry

    config = app_config(argv)
    tools = ToolRegistry(config)  # the only place tools are instantiated: always this machine
    print(f"tools: {', '.join(p.name for p in tools)}")
    log = EventLog()
    ChatServer(Assistant(config, tools, log), log).run(config.host, config.port, config.web)


def model(argv):
    from app.server.api import ModelServer
    from app.server.assistant import Assistant, EventLog
    from app.tools import ToolCatalog

    config = app_config(argv)
    log = EventLog()
    chat = Assistant.load_chat(config, ToolCatalog(config), log)
    ModelServer(chat, log).run(config.host, config.model_port)


def finetune(argv):
    from app.server.assistant import Assistant, EventLog
    from app.tools import ToolCatalog

    config = app_config(argv)
    config.train = True
    Assistant.load_chat(config, ToolCatalog(config), EventLog())


def prepare(argv):
    from model.jern.data import ChatDatasetPreparer

    ChatDatasetPreparer(os.path.join(ROOT, DATA_DIR, DATASET)).run()


def train(argv):
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


def colab(argv):
    from dataclasses import fields

    from model.jern.colab import ColabConfig, ColabRunner, RemoteError
    from model.jern.config import configure

    split = argv.index('--') if '--' in argv else len(argv)
    options, command = argv[:split], argv[split + 1:] or ['train']
    # Colab options written after `--` still belong to the runner, not to the remote command.
    keys = {f.name for f in fields(ColabConfig)}
    misplaced = [a for a in command[1:] if a.startswith('--') and a[2:].split('=', 1)[0] in keys]
    command = [a for a in command if a not in misplaced]
    config = configure(ColabConfig, options + misplaced)
    try:
        ColabRunner(config, ROOT).run(command)
    except RemoteError as error:
        sys.exit(f"colab: {error}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ('-h', '--help', 'help'):
        print(USAGE)
        return
    command = argv.pop(0) if argv and argv[0] in COMMANDS else 'serve'
    globals()[command](argv)


if __name__ == '__main__':
    main()
