import yaml
from pathlib import Path
from typing import Union

def load_config(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)