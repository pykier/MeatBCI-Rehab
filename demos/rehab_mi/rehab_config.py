import json
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "experiment_config.json"


def repo_root():
    return Path(__file__).resolve().parents[2]


def load_config(path=None):
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    config["_config_path"] = str(config_path.resolve())
    return config


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def data_root(config):
    return resolve_repo_path(config["data_root"])


def model_root(config):
    return resolve_repo_path(config["model_root"])


def session_dir(config, subject=None, session=None):
    subject = subject or config["subject"]
    session = session or config["session"]
    return data_root(config) / subject / session


def selected_model_path(config, subject=None):
    subject = subject or config["subject"]
    selected = Path(config["selected_model"])
    if selected.is_absolute():
        return selected
    if selected.parent != Path("."):
        return repo_root() / selected
    return model_root(config) / subject / selected
