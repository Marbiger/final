
import os
import json
import hashlib
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import torch
from datetime import datetime


class LLMCache:
    
    def __init__(self, cache_dir: str = '.cache/llm_predictions'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache = {}
    
    def _get_cache_key(self, bbox1: Tuple[float, ...], bbox2: Tuple[float, ...]) -> str:

        key_str = f"{bbox1}_{bbox2}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, bbox1: Tuple[float, ...], bbox2: Tuple[float, ...]) -> Optional[Dict]:
  
        key = self._get_cache_key(bbox1, bbox2)
        
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.memory_cache[key] = data
                    return data
            except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
                print(f"Warning: Cache file {cache_path} is corrupted ({e}), removing it.")
                os.remove(cache_path)
        
        return None
    
    def set(self, bbox1: Tuple[float, ...], bbox2: Tuple[float, ...], result: Dict):

        key = self._get_cache_key(bbox1, bbox2)
        self.memory_cache[key] = result
        
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    def clear(self):

        self.memory_cache.clear()
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)


class LLMLogger:

    
    def __init__(self, log_dir: str = '.logs/llm_calls'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'llm_calls_{self.session_id}.jsonl')
    
    def log(self, prompt: str, response: str, duration_ms: float, metadata: Dict = None):

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'prompt': prompt,
            'response': response,
            'duration_ms': duration_ms,
            'metadata': metadata or {}
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_stats(self) -> Dict:

        import glob
        total_calls = 0
        total_duration = 0
        
        for log_file in glob.glob(os.path.join(self.log_dir, '*.jsonl')):
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        total_calls += 1
                        total_duration += data.get('duration_ms', 0)
        
        return {
            'total_calls': total_calls,
            'total_duration_ms': total_duration,
            'avg_duration_ms': total_duration / total_calls if total_calls > 0 else 0
        }