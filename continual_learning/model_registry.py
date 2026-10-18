"""Append-only temporal model registry; baseline directories never writable here."""
import json
from pathlib import Path
class ModelRegistry:
    def __init__(self,path):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.entries=json.loads((self.path/'registry.json').read_text()) if (self.path/'registry.json').exists() else []
    def allocate(self,version):
        dest=self.path/f'version_{version:03d}'
        if dest.exists():raise FileExistsError('Model version already exists')
        dest.mkdir();return dest
    def register(self,path,deployment_time,fit_max_completion):
        if fit_max_completion>deployment_time:raise ValueError('Future-verified data in model deployment')
        self.entries.append(dict(path=str(path),deployment_time=deployment_time,maximum_verified_completion=fit_max_completion))
        (self.path/'registry.json').write_text(json.dumps(self.entries,indent=2))
