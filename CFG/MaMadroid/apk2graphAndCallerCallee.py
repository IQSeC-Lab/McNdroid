#coding=utf-8
#  python apk2graphAndCallerCallee.py <sha256_hash>

import os
import sys
import time

import requests

import abstractGraph
import apk2graph
import gml2txt

# ------------------------
# AndroZoo download config
# ------------------------
ANDROZOO_URL = "https://androzoo.uni.lu/api/download"
API_KEY = os.environ.get("API_KEY")

APK_DIR = os.path.join(os.getcwd(), "apk")
GML_DIR = os.path.join(os.getcwd(), "gml")
TXT_DIR = os.path.join(os.getcwd(), "graphs", "Trial1")

for d in [APK_DIR, GML_DIR, TXT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

def download_apk(apk_hash):
    """
    Download APK from AndroZoo into ./apk/<hash>.apk and return the local path.
    Returns None on failure.
    """
    if not API_KEY:
        print("API_KEY environment variable not set; cannot download from AndroZoo")
        return None

    apk_path = os.path.join(APK_DIR, apk_hash + ".apk")

    # Reuse cached file if it already exists
    if os.path.exists(apk_path):
        print("APK already cached at %s" % apk_path)
        return apk_path

    print("Downloading APK %s from AndroZoo..." % apk_hash)

    try:
        r = requests.get(
            ANDROZOO_URL,
            params={"apikey": API_KEY, "sha256": apk_hash},
            stream=True,
            timeout=120
        )
    except Exception as e:
        print("Request to AndroZoo failed: %s" % e)
        return None

    if r.status_code != 200:
        print("Download failed (status %d): %s" % (r.status_code, r.text[:200]))
        return None

    with open(apk_path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

    print("Saved APK to %s" % apk_path)
    return apk_path

def main():
    # Optional: SHA256 hash as first CLI argument
    apk_hash = None
    if len(sys.argv) >= 2:
        apk_hash = sys.argv[1].strip()

    # If a hash is provided, download that APK first
    if apk_hash:
        apk_path = download_apk(apk_hash)
        if not apk_path:
            sys.exit(1)

    apkfile = APK_DIR + "/"   # apk dir
    gmlfile = GML_DIR + "/"
    txtfile = TXT_DIR + "/"

    num = 0  # num to count

    '''
    apk to gml (call graph via androguard)
    '''
    apk_filenames = os.listdir(apkfile)

    # If we downloaded a specific hash, only process that file
    if apk_hash:
        expected_name = apk_hash + ".apk"
        apk_filenames = [fn for fn in apk_filenames if fn == expected_name]

    for filename in apk_filenames:
        try:
            if filename.endswith(".apk"):
                gmlpath = gmlfile + filename.rpartition(".")[0] + ".gml"
            else:
                gmlpath = gmlfile + filename + ".gml"

            full_apk_path = apkfile + filename
            apk2graph.extractcg(full_apk_path, gmlpath)
        except Exception as e:
            print("%s to gml has some error: %s" % (filename, e))
        else:
            print("%s to gml done" % filename)

    print("<----------------------apk to gml done------------------------->")

    '''
    gml to txt (txt file for abstractGraph.py)
    '''
    for gmlname in os.listdir(gmlfile):
        try:
            storepath = txtfile + gmlname.rpartition(".")[0] + ".txt"
            full_gml = gmlfile + gmlname
            g, edgelist = gml2txt.gml2graph(full_gml)
            gml2txt.caller2callee(edgelist, g.vs, storepath)
        except Exception as e:
            print("%s to txt has some error: %s" % (gmlname, e))
        else:
            print("%s to txt done" % gmlname)

    print("<----------------------gml to txt done------------------------->")

    '''
    abstract graph
    '''
    logfile = os.getcwd() + "/log.txt"

    with open(logfile, 'w') as log:
        for txtname in os.listdir(txtfile):
            txtpath = txtfile + txtname
            _app_dir = os.getcwd()
            abstractGraph._preprocess_graph(txtpath, _app_dir)  # txt path and pwd
            log.write(txtname.rpartition(".")[0] + ".apk is abstracted\n")
            num += 1
            log.write(str(num) + " apks have done\n")

    print("<----------------------abstract done--------------------------->")

if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    print("time cost:", time_end - time_start, "s")
