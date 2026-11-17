"""Source hashes and per-run snapshots for a stage without its own git repository."""
import hashlib
from pathlib import Path
import shutil


def source_manifest(project):
    files = [*sorted((project / "code").rglob("*.py")), *sorted((project / "scripts").glob("*.py")),
             project / "README.md", project / "requirements.txt", project / "BASE_SOURCE.json"]
    return {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def snapshot(project, output, expected):
    if source_manifest(project) != expected:
        raise RuntimeError("Source files changed before snapshot")
    destination = output / "source"
    destination.mkdir()
    for name in expected:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected[name]:
            raise RuntimeError(f"Source changed while copying: {name}")
