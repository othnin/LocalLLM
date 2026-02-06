import yaml
from pathlib import Path


def load_assumptions(intent: str):
    path = Path("assumptions") / f"{intent}.yaml"
    if not path.exists():
        return []

    with open(path) as f:
        data = yaml.safe_load(f)

    return data.get("assumptions", [])
