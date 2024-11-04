from typing import List, Dict
import json

def read_jsonl(path: str) -> List[Dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]