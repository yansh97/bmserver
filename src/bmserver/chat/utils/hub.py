from fnmatch import fnmatch
from pathlib import Path

from bmhub.backend import HubBackend
from bmhub.schema import ModelInfo

from bmserver.chat.schema import MODELS_JSONL, Model, ModelSpec


def search_models(
    *, hub: HubBackend, pattern: str | None, local_dir: Path | None
) -> list[Model]:
    specs: list[ModelSpec] = [
        ModelSpec.model_validate_json(json_data=line)
        for line in MODELS_JSONL.read_text(encoding="utf-8").splitlines()
    ]
    specs = [spec for spec in specs if hub in spec.repo_id]
    if pattern is not None:
        specs = [spec for spec in specs if fnmatch(name=spec.name, pat=pattern)]
    infos: dict[str, ModelInfo] = {
        m.id: m for m in hub.list_models(pattern=None, local_dir=local_dir)
    }
    return [Model(spec=spec, info=infos.get(spec.repo_id[hub])) for spec in specs]


def find_model(*, hub: HubBackend, name: str, local_dir: Path | None) -> Model:
    models: dict[str, Model] = {
        model.spec.name: model
        for model in search_models(hub=hub, pattern=None, local_dir=local_dir)
    }
    if name not in models:
        raise ValueError(f"No supported chat model found: {name=}.")
    return models[name]
