import json
from pathlib import Path


def load_json(file_path: Path) -> dict:
    """
    Load and return JSON data from a file.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict, file_path: Path):
    """
    Save dictionary as formatted JSON.
    """

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def get_latest_file(folder: Path) -> Path:
    """
    Return the latest JSON file in a folder.
    """

    json_files = list(folder.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {folder}")

    return max(json_files, key=lambda file: file.stat().st_mtime)