# app/services/model_manager.py
import os
import json
from pathlib import Path
import joblib
import hashlib

class ModelManager:
    """
    Manages model versions inside models/:
      - saves new versions under models/versions/<version>/
      - maintains models/active -> symlink or file pointing to active version
    """
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.versions_dir = self.models_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.active_file = self.models_dir / "active_version.json"

    def _version_path(self, ver):
        return self.versions_dir / ver

    def list_versions(self):
        out = []
        for p in sorted(self.versions_dir.iterdir(), key=lambda x: x.name):
            if p.is_dir():
                meta_file = p / "meta.json"
                meta = {}
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text())
                out.append({"version": p.name, "meta": meta, "path": str(p)})
        active = self.get_active_version()
        return {"versions": out, "active": active}

    def save_new_version(self, raw_bytes: bytes, version: str, filename: str = "model.joblib"):
        path = self._version_path(version)
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / filename
        model_path.write_bytes(raw_bytes)
        # save basic metadata
        meta = {
            "filename": filename,
            "created_at": __import__("datetime").datetime.utcnow().isoformat(),
            "sha256": hashlib.sha256(raw_bytes).hexdigest()
        }
        (path / "meta.json").write_text(json.dumps(meta))
        return str(path)

    def activate_version(self, version: str):
        path = self._version_path(version)
        if not path.exists():
            return False
        active_data = {"active": version, "activated_at": __import__("datetime").datetime.utcnow().isoformat()}
        self.active_file.write_text(json.dumps(active_data))
        return True

    def get_active_version(self):
        if not self.active_file.exists():
            return None
        try:
            data = json.loads(self.active_file.read_text())
            return data.get("active")
        except Exception:
            return None

    def get_active_model_path(self):
        active = self.get_active_version()
        if not active:
            return None
        p = self._version_path(active)
        # pick model.joblib or first joblib in directory
        candidates = list(p.glob("*.joblib"))
        if candidates:
            return str(candidates[0])
        return None
