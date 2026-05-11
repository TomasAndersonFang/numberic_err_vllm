import json
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """Load a YAML configuration file and return it as a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return a list of dictionaries."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: list[dict]) -> None:
    """Write a list of dictionaries to a JSONL file, one JSON object per line."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
