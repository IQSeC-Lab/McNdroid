# coding:utf-8
"""
Run apk2graphAndCallerCallee.py for each hash in a CSV file.

Assumptions:
  - CSV has a header row.
  - The first column in each row is the SHA256 hash.
Usage:
  python run_from_csv.py hashes.csv
"""

import sys
import os
import csv
import subprocess

SCRIPT_NAME = "apk2graphAndCallerCallee.py"

def run_for_hash(apk_hash):
    apk_hash = apk_hash.strip()
    if not apk_hash:
        return

    cmd = ["python", SCRIPT_NAME, apk_hash]
    print("Running: %s" % " ".join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        print("  -> FAILED for hash %s (exit code %d)" % (apk_hash, ret))
    else:
        print("  -> DONE for hash %s" % apk_hash)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_from_csv.py <csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print("CSV file not found: %s" % csv_path)
        sys.exit(1)

    with open(csv_path, "r") as f:
        reader = csv.reader(f)

        # Skip header row
        try:
            header = next(reader)
            print("Header:", header)
        except StopIteration:
            print("CSV file is empty.")
            return

        # Process remaining rows
        for row in reader:
            if not row:
                continue
            apk_hash = row[0]
            run_for_hash(apk_hash)

if __name__ == "__main__":
    main()
