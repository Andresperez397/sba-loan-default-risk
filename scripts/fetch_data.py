"""Download the SBA 7(a) loan-level FOIA files into data/raw/ and pin them by SHA-256.

SBA republishes these files every quarter under new names, so this project is pinned to the
snapshot "as of June 30, 2026". The first run writes data/manifest.csv with each file's size and
SHA-256; later runs verify every file and stop if anything differs.
"""

from __future__ import annotations

import csv
import hashlib
import ssl
import sys
import urllib.request
from datetime import date
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"
BASE = "https://data.sba.gov/sites/default/files/uploaded_resources"
FILES = [
    "7a_504_foia_data_dictionary.xlsx",
    "FOIA_7a_FY2000_FY2009_asof_260630.csv",
    "FOIA_7a_FY2010_FY2019_asof_260630.csv",
    "FOIA_7a_FY2020_Present_asof_260630.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    ctx = ssl.create_default_context(cafile=certifi.where())
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, context=ctx, timeout=600) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    with open(tmp, "rb") as f:
        head = f.read(64)
    if head.lstrip().lower().startswith((b"<!doctype", b"<html")):  # an error page, not data
        tmp.unlink()
        raise RuntimeError(f"got an HTML page instead of data: {url}")
    tmp.rename(dest)


PRIME_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DPRIME"


def prime_fingerprint(path: Path) -> str:
    """FRED appends a row every day, so pin only the dates this project uses (FY2009-FY2021)."""
    lines = [ln for ln in path.read_text().splitlines()[1:] if "2008-10-01" <= ln[:10] <= "2021-09-30"]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in FILES:
        dest = RAW / name
        if not dest.exists():
            print("downloading", name, flush=True)
            download(f"{BASE}/{name}", dest)
        rows.append(
            {"name": name, "url": f"{BASE}/{name}", "bytes": dest.stat().st_size, "sha256": sha256(dest)}
        )
    prime = RAW / "DPRIME.csv"
    if not prime.exists():
        print("downloading DPRIME.csv", flush=True)
        download(PRIME_URL, prime)
    rows.append({"name": "DPRIME.csv", "url": PRIME_URL, "bytes": "", "sha256": prime_fingerprint(prime)})
    if not MANIFEST.exists() or "--repin" in sys.argv:
        for r in rows:
            r["pinned_on"] = date.today().isoformat()
        with open(MANIFEST, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(rows)} files)")
        return
    with open(MANIFEST) as f:
        pinned = {r["name"]: r["sha256"] for r in csv.DictReader(f)}
    bad = [r["name"] for r in rows if pinned.get(r["name"]) != r["sha256"]]
    if bad:
        raise SystemExit(f"files differ from data/manifest.csv: {bad}")
    print(f"all {len(rows)} files match data/manifest.csv")


if __name__ == "__main__":
    main()
