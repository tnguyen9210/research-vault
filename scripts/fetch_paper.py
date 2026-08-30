#!/usr/bin/env python3
"""Mirror Zotero PDFs into a citekey-named folder.

Usage:
    fetch_paper.py CITEKEY [CITEKEY ...]
    fetch_paper.py --sync-all [--prune]

Resolves papers by Better BibTeX citekey (Zotero 8 native
citationKey field, read via the Zotero web API) and saves
each PDF as <dest>/<citekey>.pdf. Never resolves paths from
library.json attachment entries -- those are machine-local.

Credentials: ~/.config/zotero/credentials with lines
ZOTERO_USER_ID=... and ZOTERO_API_KEY=... (or the same names
as environment variables, which take precedence).

Destination: --dest, else $ZOTERO_PAPERS_DIR, else the UA
group-storage default if it exists on this machine.

A cache of citekey -> attachment mappings is kept in
<dest>/.zotero-index.json; --refresh rebuilds it.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://api.zotero.org"
CRED_PATH = os.path.expanduser("~/.config/zotero/credentials")
DEFAULT_DEST = "/groups/chichengz/tnn/zoterostorage"
INDEX_NAME = ".zotero-index.json"
PAGE = 100
RETRIES = 5


def load_creds():
    uid = os.environ.get("ZOTERO_USER_ID")
    key = os.environ.get("ZOTERO_API_KEY")
    if uid and key:
        return uid, key
    if os.path.exists(CRED_PATH):
        with open(CRED_PATH) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("ZOTERO_USER_ID="):
                    uid = uid or line.split("=", 1)[1]
                elif line.startswith("ZOTERO_API_KEY="):
                    key = key or line.split("=", 1)[1]
    if not (uid and key):
        sys.exit(f"error: Zotero credentials not found "
                 f"(env or {CRED_PATH})")
    return uid, key


def api_request(url, key, binary=False):
    """GET with retry/backoff; honors Retry-After on 429/5xx."""
    for attempt in range(RETRIES):
        req = urllib.request.Request(
            url, headers={"Zotero-API-Key": key})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
                return body if binary else json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and \
                    attempt < RETRIES - 1:
                wait = int(e.headers.get("Retry-After", 0) or 0)
                time.sleep(max(wait, 2 ** attempt))
                continue
            raise
        except urllib.error.URLError:
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def get_paged(uid, key, path, extra=""):
    out, start = [], 0
    while True:
        url = (f"{API}/users/{uid}/{path}?limit={PAGE}"
               f"&start={start}&format=json{extra}")
        batch = api_request(url, key)
        out += batch
        if len(batch) < PAGE:
            return out
        start += PAGE


def build_index(uid, key):
    """citekey -> {item, att, md5, filename}; skips keyless
    items and papers with no stored PDF attachment."""
    tops = get_paged(uid, key, "items/top")
    atts = get_paged(uid, key, "items",
                     "&itemType=attachment")
    by_parent = {}
    for a in atts:
        d = a["data"]
        if d.get("contentType") != "application/pdf":
            continue
        if not d.get("linkMode", "").startswith("imported"):
            continue
        parent = d.get("parentItem")
        if parent:
            by_parent.setdefault(parent, []).append(d)
    papers, no_pdf = {}, []
    for it in tops:
        d = it["data"]
        ck = d.get("citationKey")
        if not ck:
            continue
        pdfs = sorted(by_parent.get(it["key"], []),
                      key=lambda x: x.get("dateAdded", ""))
        if not pdfs:
            no_pdf.append(ck)
            continue
        papers[ck] = {"item": it["key"],
                      "att": pdfs[0]["key"],
                      "md5": pdfs[0].get("md5", ""),
                      "filename": pdfs[0].get("filename", "")}
    return {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "papers": papers, "no_pdf": sorted(no_pdf)}


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(uid, key, dest, ck, entry):
    """Returns 'fetched' | 'skipped' | error string."""
    target = f"{dest}/{ck}.pdf"
    if os.path.exists(target) and entry["md5"] and \
            md5_file(target) == entry["md5"]:
        return "skipped"
    url = f"{API}/users/{uid}/items/{entry['att']}/file"
    try:
        blob = api_request(url, key, binary=True)
    except Exception as e:
        return f"download failed: {e}"
    tmp = target + ".part"
    with open(tmp, "wb") as fh:
        fh.write(blob)
    os.replace(tmp, target)
    return "fetched"


def main():
    ap = argparse.ArgumentParser(
        description="Mirror Zotero PDFs by citekey.")
    ap.add_argument("citekeys", nargs="*")
    ap.add_argument("--sync-all", action="store_true",
                    help="mirror every paper with a PDF")
    ap.add_argument("--prune", action="store_true",
                    help="with --sync-all: delete PDFs whose "
                         "citekey left the library")
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild the citekey index")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel downloads "
                         "(default min(8, n); API-polite)")
    ap.add_argument("--dest", default=None)
    args = ap.parse_args()
    if bool(args.citekeys) == args.sync_all:
        ap.error("give citekeys OR --sync-all")

    dest = (args.dest or os.environ.get("ZOTERO_PAPERS_DIR")
            or DEFAULT_DEST)
    if not os.path.isdir(dest):
        sys.exit(f"error: dest dir not found: {dest} "
                 f"(use --dest or $ZOTERO_PAPERS_DIR)")
    uid, key = load_creds()

    index_path = f"{dest}/{INDEX_NAME}"
    index = None
    if not args.refresh and not args.sync_all and \
            os.path.exists(index_path):
        with open(index_path) as fh:
            index = json.load(fh)
        if any(ck not in index["papers"]
               for ck in args.citekeys):
            index = None  # unknown key: index may be stale
    if index is None:
        print("building index from Zotero API ...")
        index = build_index(uid, key)
        with open(index_path, "w") as fh:
            json.dump(index, fh, indent=1)
        print(f"index: {len(index['papers'])} papers with "
              f"PDFs, {len(index['no_pdf'])} without")

    papers = index["papers"]
    if args.sync_all:
        targets = sorted(papers)
    else:
        missing = [c for c in args.citekeys
                   if c not in papers]
        for c in missing:
            note = (" (item has no stored PDF)"
                    if c in index["no_pdf"] else "")
            print(f"error: unknown citekey: {c}{note}")
        targets = [c for c in args.citekeys if c in papers]

    jobs = args.jobs or min(8, max(1, len(targets)))
    counts = {"fetched": 0, "skipped": 0}
    failed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(fetch_one, uid, key, dest, ck,
                            papers[ck]): ck
                for ck in targets}
        for fut in as_completed(futs):
            ck, res = futs[fut], fut.result()
            if res in counts:
                counts[res] += 1
                if not args.sync_all:
                    print(f"{res:8s} {dest}/{ck}.pdf")
            else:
                failed.append((ck, res))
                print(f"FAILED   {ck}: {res}")

    if args.sync_all and args.prune:
        keep = {f"{c}.pdf" for c in papers}
        for name in sorted(os.listdir(dest)):
            if name.endswith(".pdf") and name not in keep:
                os.remove(f"{dest}/{name}")
                print(f"pruned   {name}")

    print(f"done: {counts['fetched']} fetched, "
          f"{counts['skipped']} up-to-date, "
          f"{len(failed)} failed"
          + (f", {len(index['no_pdf'])} items without PDF"
             if args.sync_all else ""))
    sys.exit(1 if (failed or (not args.sync_all and missing))
             else 0)


if __name__ == "__main__":
    main()
