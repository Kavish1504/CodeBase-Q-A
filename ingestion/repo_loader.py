from __future__ import annotations
from pathlib import Path
import os
from git import Repo,InvalidGitRepositoryError

from loguru import logger
import shutil
from typing import Iterator

SUPPORTED_EXNTENSIONS:set[str]={
    # Python
    ".py",
    # JavaScript / TypeScript
    ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    # Java / Kotlin
    ".java", ".kt",
    # C / C++
    ".c", ".cpp", ".h", ".hpp",
    # Go
    ".go",
    # Rust
    ".rs",
    # Ruby
    ".rb",
    # C#
    ".cs",
    # PHP
    ".php",
    # Swift
    ".swift",
    # Shell
    ".sh", ".bash",
}

SKIP_DIR:set[str]={
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "env", "dist", "build", ".mypy_cache", ".pytest_cache",
    "*.egg-info", ".tox", "coverage", ".next", ".nuxt",
}

def _should_skip(path:Path):
    return any(part in SKIP_DIR for part in path.parts)

def clone_repo(repo_url:str,base_url:str="./cloned_repos")->Path:
    repo_name=repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    clone_path=Path(base_url)/repo_name
    clone_path.parent.mkdir(parents=True,exist_ok=True)
    if clone_path.exists():
        try:
            logger.info(f"Repo already cloned at {clone_path}. Pulling latest…")
            existing=Repo(clone_path)
            existing.remotes.origin.pull()
            return clone_path
        except InvalidGitRepositoryError:
            logger.info("Directory exists but is not a git repo — re-cloning.")
            shutil.rmtree(clone_path)

    logger.info(f"Cloning {repo_url} → {clone_path}")
    Repo.clone_from(repo_url,clone_path,depth=1)
    return clone_path

def iter_source_files(repo_path:Path)->Iterator[dict]:
    for abs_path in repo_path.rglob("*"):
        if not abs_path.is_file():
            continue
        if _should_skip(abs_path.relative_to(repo_path)):
            continue
        if abs_path.suffix not in SUPPORTED_EXNTENSIONS:
            continue
        
        size=abs_path.stat().st_size
        if size > 500_000:
            logger.debug(f"Skipping large file ({size} bytes): {abs_path}")
            continue

        try:
            content=abs_path.read_text(encoding="utf-8",errors="ignore")
        except Exception as e:
            logger.info(f"Could not read {abs_path}: {e}")
            continue

        yield {
            "file_path": str(abs_path.relative_to(repo_path)),
            "abs_path": abs_path,
            "content": content,
            "language": abs_path.suffix.lstrip("."),
            "size_bytes": size,
        }

def load_repo(repo_url:str,base_url:str="./cloned_repos")->list[dict]:
    clone_path=clone_repo(repo_url,base_url)
    files=list(iter_source_files(clone_path))
    logger.info(f"Loaded {len(files)} source files from {clone_path}")
    return files






    
