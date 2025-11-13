# src/mist_statinf/utils/downloads.py
from __future__ import annotations
import hashlib, os, shutil, subprocess, sys
from pathlib import Path
from typing import Dict, Tuple, Optional
from urllib.request import urlopen

BUFFER = 1024 * 1024  # 1 MB chunks

PRESETS: Dict[str, Dict[str, Tuple[str, Optional[str]]]] = {
    "m_test_imd": {
        "M_test_imd_dataset.pkl": ("https://zenodo.org/records/14452998/files/AOD_Districts.csv", None),
        "M_test_imd_meta_info.json":   ("https://zenodo.org/records/14452998/files/AOD_Districts.csv",   None),
        "M_test_imd_config.yaml":  ("https://zenodo.org/records/14452998/files/AOD_Districts.csv",  None),
    },
    "m_test_oomd": {
        "M_test_oomd_dataset.pkl": ("https://zenodo.org/records/14452998/files/AOD_Districts.csv", None),
        "M_test_oomd_meta_info.json":   ("https://zenodo.org/records/14452998/files/AOD_Districts.csv",   None),
        "M_test_oomd_config.yaml":  ("https://zenodo.org/records/14452998/files/AOD_Districts.csv",  None),
    },
    "m_test_extended_imd": {
        "M_test_extended_imd_dataset.pkl": ("https://zenodo.org/records/14452998/files/AOD_Districts.csv", None),
        "M_test_extended_imd_meta_info.json":   ("https://zenodo.org/records/14452998/files/AOD_Districts.csv",   None),
        "M_test_extended_imd_config.yaml":  ("https://zenodo.org/records/14452998/files/AOD_Districts.csv",  None),
    },
    "m_test_extended_oomd": {
        "M_test_extended_oomd_dataset.pkl": ("https://zenodo.org/records/14452998/files/AOD_Districts.csv", None),
        "M_test_extended_oomd_meta_info.json":   ("https://zenodo.org/records/14452998/files/AOD_Districts.csv",   None),
        "M_test_extended_oomd_config.yaml":  ("https://zenodo.org/records/14452998/files/AOD_Districts.csv",  None),
    },
    "m_train": {
        "M_train_dataset.pkl": ("https://zenodo.org/records/14452998/files/AOD_Districts.csv", None),
        "M_train_meta_info.json":   ("https://zenodo.org/records/14452998/files/AOD_Districts.csv",   None),
        "M_train_config.yaml":  ("https://zenodo.org/records/14452998/files/AOD_Districts.csv",  None),
    },
}

def _has_wget() -> bool:
    return shutil.which("wget") is not None

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(BUFFER), b""):
            h.update(chunk)
    return h.hexdigest()

def _download_with_wget(url: str, out_path: Path, quiet: bool) -> None:
    args = ["wget", "-O", str(out_path), "-c", url]  # -c resume
    if quiet:
        args.insert(1, "-q")
        args.insert(1, "--show-progress")
    subprocess.run(args, check=True)

def _download_with_python(url: str, out_path: Path, quiet: bool) -> None:
    with urlopen(url) as r, out_path.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        read = 0
        while True:
            chunk = r.read(BUFFER)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if not quiet and total:
                pct = 100 * read / total
                sys.stderr.write(f"\rDownloading {out_path.name}: {pct:5.1f}%")
                sys.stderr.flush()
        if not quiet and total:
            sys.stderr.write("\n")

def fetch_files(
    items: Dict[str, Tuple[str, Optional[str]]],
    target_dir: Path,
    quiet: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Download multiple files into target_dir.
    Each item: name -> (url, sha256 or None).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, sha256sum) in items.items():
        out_path = target_dir / name
        if out_path.exists() and not overwrite:
            if sha256sum:
                actual = _sha256(out_path)
                if actual != sha256sum:
                    out_path.unlink()
                else:
                    if not quiet:
                        print(f"[skip] {name} (exists, checksum ok)")
                    continue
            else:
                if not quiet:
                    print(f"[skip] {name} (exists)")
                continue

        if not quiet:
            print(f"[get ] {name} ← {url}")
        try:
            if _has_wget():
                _download_with_wget(url, out_path, quiet)
            else:
                _download_with_python(url, out_path, quiet)
        except Exception as e:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download {url}: {e}") from e

        if sha256sum:
            actual = _sha256(out_path)
            if actual != sha256sum:
                out_path.unlink(missing_ok=True)
                raise RuntimeError(f"Checksum mismatch for {name}. Expected {sha256sum}, got {actual}")

def preset_bundle(name: str) -> Dict[str, Tuple[str, Optional[str]]]:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
    return PRESETS[name]
