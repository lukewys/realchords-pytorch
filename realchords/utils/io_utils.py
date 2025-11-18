import json
from typing import Any


class JSONLIndexer:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.offsets = []
        self._build_index()

    def _build_index(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line.encode("utf-8"))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Any:
        return self(idx)

    def __call__(self, idx: int) -> Any:
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError(
                f"Index {idx} out of range for file with {len(self)} lines"
            )
        with open(self.filepath, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            return json.loads(line)


def save_jsonl(data: Any, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
