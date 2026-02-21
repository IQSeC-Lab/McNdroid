#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AndroZoo (download) -> Androguard (cg -> .gml)

Outputs (ALWAYS in the same folder as this script):
  ./apks/<year>/<sha256>.apk
  ./gml/<year>/<sha256>.gml
  ./logs/<year>/download.log
  ./logs/<year>/extracted_gml.log
  ./logs/<year>/apk_timing.log

Logs:
- download.log          : sha256 only (successful downloads/verified-existing), one per line, deduped
- extracted_gml.log     : sha256 only (successful GML extractions), one per line, deduped
- apk_timing.log        : TSV per successfully extracted APK:
                           sha256<TAB>download_sec<TAB>gml_sec<TAB>total_sec
                         deduped by sha256

Pipeline:
- Submits downloads for --download-minutes (default 30). In-flight downloads finish.
- As soon as an APK is verified, it is queued for GML extraction.
- Extraction uses ProcessPoolExecutor.

Note:
- If download is "skipped because already in download.log", download_sec=0.0.
- If you want skipped GML (existing .gml) to count as processed/logged, set --count-skipped-gml.
"""

import argparse
import concurrent.futures as cf
import hashlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_PROCESSES = 10
DEFAULT_MAX_DOWNLOAD_CONCURRENT = 8
ANDROZOO_URL = "https://androzoo.uni.lu/api/download"

# ============================================================
# Hardcode AndroZoo API keys here
# ============================================================
API_KEYS = [
    "PUT_KEY_1_HERE",
    "PUT_KEY_2_HERE",
    # "PUT_KEY_3_HERE",
]


# ----------------------------
# Paths (relative to script dir)
# ----------------------------

def script_root() -> Path:
    return Path(__file__).resolve().parent


# ----------------------------
# Utilities
# ----------------------------

def normalize_sha256(s: str) -> str:
    return s.strip().lower()


def safe_append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def load_done_set_simple(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            h = line.strip().lower()
            if len(h) == 64:
                done.add(h)
    return done


def load_done_set_timing(path: Path) -> set[str]:
    """
    timing log format: sha256<TAB>download_sec<TAB>gml_sec<TAB>total_sec
    """
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                h = parts[0].strip().lower()
                if len(h) == 64:
                    done.add(h)
    return done


def read_hashes(hash_file: Path) -> list[str]:
    hashes: list[str] = []
    with hash_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            h = normalize_sha256(line)
            if not h:
                continue
            if len(h) == 64 and all(c in "0123456789abcdef" for c in h):
                hashes.append(h)

    # preserve order, dedupe
    seen = set()
    uniq: list[str] = []
    for h in hashes:
        if h in seen:
            continue
        seen.add(h)
        uniq.append(h)
    return uniq


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pick_key_index(stable_index: int, attempt: int, key_count: int) -> int:
    base = stable_index % key_count
    return (base + attempt) % key_count


# ----------------------------
# Androguard extraction
# ----------------------------

def run_androguard_cg(
    androguard_bin: str,
    apk_path: Path,
    gml_out: Path,
    timeout_sec: int,
    atomic: bool = True,
) -> tuple[bool, str]:
    gml_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = gml_out.with_suffix(gml_out.suffix + ".tmp") if atomic else gml_out

    cmd = [androguard_bin, "cg", str(apk_path), "-o", str(tmp_out)]
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        if p.returncode == 0 and tmp_out.exists() and tmp_out.stat().st_size > 0:
            if atomic:
                tmp_out.replace(gml_out)
            return True, ""
        err = (p.stderr or p.stdout or f"exit={p.returncode}")[:2000]
        if atomic and tmp_out.exists():
            try:
                tmp_out.unlink()
            except Exception:
                pass
        return False, err
    except subprocess.TimeoutExpired:
        if atomic and tmp_out.exists():
            try:
                tmp_out.unlink()
            except Exception:
                pass
        return False, f"TIMEOUT after {timeout_sec}s"
    except Exception as e:
        if atomic and tmp_out.exists():
            try:
                tmp_out.unlink()
            except Exception:
                pass
        return False, repr(e)


def process_one(args):
    androguard_bin, apk_path, out_dir, timeout_sec, skip_existing = args
    t0 = time.time()

    sha = apk_path.stem.lower()
    gml_out = out_dir / f"{sha}.gml"

    if skip_existing and gml_out.exists() and gml_out.stat().st_size > 0:
        return {
            "apk_sha256": sha,
            "status": "skipped",
            "elapsed_sec": round(time.time() - t0, 3),
        }

    ok, info = run_androguard_cg(
        androguard_bin=androguard_bin,
        apk_path=apk_path,
        gml_out=gml_out,
        timeout_sec=timeout_sec,
        atomic=True,
    )

    return {
        "apk_sha256": sha,
        "status": "ok" if ok else "fail",
        "elapsed_sec": round(time.time() - t0, 3),
        "error": "" if ok else (info or "")[:2000],
    }


# ----------------------------
# AndroZoo download
# ----------------------------

@dataclass
class DownloadResult:
    sha256: str
    status: str      # ok | skipped | fail
    apk_path: str
    elapsed_sec: float  # download/verify wall time (0 for skipped-by-log)


def download_one_apk(
    sha256: str,
    sha_index: int,
    out_dir: Path,
    max_apk_mb: int,
    timeout_sec: int,
    download_done_set: set[str],
    download_log: Path,
    retries: int,
) -> DownloadResult:
    t0 = time.time()
    sha256 = normalize_sha256(sha256)
    out_dir.mkdir(parents=True, exist_ok=True)
    apk_path = out_dir / f"{sha256}.apk"

    # Already recorded as successful download
    if sha256 in download_done_set:
        return DownloadResult(
            sha256=sha256,
            status="skipped",
            apk_path=str(apk_path) if apk_path.exists() else "",
            elapsed_sec=0.0,
        )

    # If file exists and is valid, record success and return ok
    if apk_path.exists() and apk_path.is_file() and apk_path.stat().st_size > 0:
        try:
            got = sha256_file(apk_path)
            if got == sha256:
                safe_append_line(download_log, sha256)
                download_done_set.add(sha256)
                return DownloadResult(
                    sha256=sha256,
                    status="ok",
                    apk_path=str(apk_path),
                    elapsed_sec=round(time.time() - t0, 3),
                )
        except Exception:
            pass

    if not API_KEYS or any((not k or "PUT_KEY" in k) for k in API_KEYS):
        return DownloadResult(sha256=sha256, status="fail", apk_path="", elapsed_sec=round(time.time() - t0, 3))

    key_count = len(API_KEYS)

    for attempt in range(retries + 1):
        key_idx = pick_key_index(sha_index, attempt, key_count)
        api_key = API_KEYS[key_idx]

        tmp_path = apk_path.with_suffix(".apk.part")
        try:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

            qs = urlencode({"apikey": api_key, "sha256": sha256})
            url = f"{ANDROZOO_URL}?{qs}"
            req = Request(url, method="GET")

            total = 0
            with urlopen(req, timeout=timeout_sec) as resp, tmp_path.open("wb") as out_f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    total += len(chunk)
                    if max_apk_mb > 0 and total > max_apk_mb * 1024 * 1024:
                        raise RuntimeError("download_exceeds_max_apk_mb")

            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("empty_download")

            got = sha256_file(tmp_path)
            if got != sha256:
                raise RuntimeError("sha256_mismatch")

            tmp_path.replace(apk_path)

            if sha256 not in download_done_set:
                safe_append_line(download_log, sha256)
                download_done_set.add(sha256)

            return DownloadResult(
                sha256=sha256,
                status="ok",
                apk_path=str(apk_path),
                elapsed_sec=round(time.time() - t0, 3),
            )

        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            continue

    return DownloadResult(sha256=sha256, status="fail", apk_path="", elapsed_sec=round(time.time() - t0, 3))


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-file", required=True, help="Text file with SHA256 per line.")
    ap.add_argument("--year", required=True, help="Year folder name under ./apks, ./gml, ./logs")

    # download controls
    ap.add_argument("--download-minutes", type=int, default=30, help="Submit downloads for N minutes.")
    ap.add_argument("--max-download-concurrent", type=int, default=DEFAULT_MAX_DOWNLOAD_CONCURRENT)
    ap.add_argument("--download-timeout-sec", type=int, default=600)
    ap.add_argument("--download-retries", type=int, default=2)
    ap.add_argument("--max-apk-mb", type=int, default=1000, help="Abort downloads larger than this (MB). 0=disable.")

    # extraction controls
    ap.add_argument("--androguard-bin", default="androguard")
    ap.add_argument("--processes", type=int, default=DEFAULT_PROCESSES)
    ap.add_argument("--timeout-sec", type=int, default=2000)
    ap.add_argument("--skip-existing-gml", action="store_true")
    ap.add_argument("--count-skipped-gml", action="store_true", help="Treat skipped existing .gml as processed for logs/timing.")
    ap.add_argument("--clean-output", action="store_true")

    args = ap.parse_args()

    if not API_KEYS or any((not k or "PUT_KEY" in k) for k in API_KEYS):
        print("ERROR: API_KEYS not configured. Edit API_KEYS[] in the script.", file=sys.stderr)
        sys.exit(2)

    hash_file = Path(args.hash_file).expanduser().resolve()
    if not hash_file.is_file():
        print(f"ERROR: hash file not found: {hash_file}", file=sys.stderr)
        sys.exit(2)

    root = script_root()

    apk_dir = (root / "apks" / str(args.year)).resolve()
    gml_dir = (root / "gml" / str(args.year)).resolve()
    log_dir = (root / "logs" / str(args.year)).resolve()

    download_log = (log_dir / "download.log").resolve()
    gml_log = (log_dir / "extracted_gml.log").resolve()
    timing_log = (log_dir / "apk_timing.log").resolve()

    apk_dir.mkdir(parents=True, exist_ok=True)
    gml_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_output:
        for p in gml_dir.glob("*.gml"):
            try:
                p.unlink()
            except Exception:
                pass

    hashes = read_hashes(hash_file)
    if not hashes:
        print("ERROR: no valid SHA256 entries in hash file", file=sys.stderr)
        sys.exit(2)

    # Load logs to dedupe
    download_done_set = load_done_set_simple(download_log)
    gml_done_set = load_done_set_simple(gml_log)
    timing_done_set = load_done_set_timing(timing_log)

    workers_proc = max(1, min(args.processes, os.cpu_count() or 1))
    workers_dl = max(1, args.max_download_concurrent)

    print(f"Hashes in input: {len(hashes)}")
    print(f"Download submission window: {args.download_minutes} minutes")
    print(f"APK dir: {apk_dir}")
    print(f"GML dir: {gml_dir}")
    print(f"Logs dir: {log_dir}")
    print(f"Download threads: {workers_dl}")
    print(f"GML processes: {workers_proc}")

    dl_ex = cf.ThreadPoolExecutor(max_workers=workers_dl)
    cg_ex = cf.ProcessPoolExecutor(max_workers=workers_proc)

    dl_futs: dict[cf.Future, tuple[str, int]] = {}
    cg_futs: dict[cf.Future, str] = {}

    # Store per-sha download times (for timing log)
    download_time_sec: dict[str, float] = {}

    t_deadline = time.time() + (args.download_minutes * 60)

    def submit_download(h: str, idx: int) -> None:
        fut = dl_ex.submit(
            download_one_apk,
            h,
            idx,
            apk_dir,
            args.max_apk_mb,
            args.download_timeout_sec,
            download_done_set,
            download_log,
            args.download_retries,
        )
        dl_futs[fut] = (h, idx)

    def submit_gml(apk_path: Path) -> None:
        job = (args.androguard_bin, apk_path, gml_dir, args.timeout_sec, args.skip_existing_gml)
        fut = cg_ex.submit(process_one, job)
        cg_futs[fut] = apk_path.stem.lower()

    # Seed downloads
    i = 0
    while i < len(hashes) and len(dl_futs) < workers_dl and time.time() < t_deadline:
        submit_download(hashes[i], i)
        i += 1

    while dl_futs or cg_futs:
        wait_set = set(dl_futs.keys()) if dl_futs else set(cg_futs.keys())
        done, _ = cf.wait(wait_set, return_when=cf.FIRST_COMPLETED)

        for fut in done:
            if fut in dl_futs:
                h, idx = dl_futs.pop(fut)
                try:
                    r: DownloadResult = fut.result()
                except Exception:
                    r = DownloadResult(sha256=h, status="fail", apk_path="", elapsed_sec=0.0)

                download_time_sec[r.sha256] = float(r.elapsed_sec)

                if r.status in ("ok", "skipped"):
                    if r.apk_path:
                        p = Path(r.apk_path)
                        if p.exists() and p.stat().st_size > 0:
                            submit_gml(p)

                # Feed more downloads while time remains
                while time.time() < t_deadline and i < len(hashes) and len(dl_futs) < workers_dl:
                    submit_download(hashes[i], i)
                    i += 1

            else:
                sha = cg_futs.pop(fut, "")
                try:
                    rec = fut.result()
                except Exception:
                    rec = {"apk_sha256": sha, "status": "fail", "elapsed_sec": 0.0}

                sha_ok = (rec.get("apk_sha256") or "").strip().lower()
                status = rec.get("status")
                gml_sec = float(rec.get("elapsed_sec", 0.0))
                dl_sec = float(download_time_sec.get(sha_ok, 0.0))
                total = dl_sec + gml_sec

                is_processed = (status == "ok") or (args.count_skipped_gml and status == "skipped")

                # extracted_gml.log: sha256 only, processed-only, deduped
                if is_processed and sha_ok and sha_ok not in gml_done_set:
                    safe_append_line(gml_log, sha_ok)
                    gml_done_set.add(sha_ok)

                # apk_timing.log: sha256<TAB>download_sec<TAB>gml_sec<TAB>total_sec, processed-only, deduped
                if is_processed and sha_ok and sha_ok not in timing_done_set:
                    safe_append_line(timing_log, f"{sha_ok}\t{dl_sec:.3f}\t{gml_sec:.3f}\t{total:.3f}")
                    timing_done_set.add(sha_ok)

    dl_ex.shutdown(wait=True, cancel_futures=False)
    cg_ex.shutdown(wait=True, cancel_futures=False)

    print("Done.")
    print(f"download.log total: {len(download_done_set)}")
    print(f"extracted_gml.log total: {len(gml_done_set)}")
    print(f"apk_timing.log total: {len(timing_done_set)}")


if __name__ == "__main__":
    main()
