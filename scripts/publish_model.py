"""
Publishes an ONNX model so the Android app picks it up without a new APK
(see app/.../ModelUpdater.kt).

Uploads the model, then manifest.json, to the public "models" bucket in the
project's Supabase Storage (creating the bucket the first time). The model
goes up first, so a phone never sees a manifest pointing at a missing file.
Before uploading, the model is checked the same way the app checks it:
exactly one (1, 50, 6) input and one (1, 50, 2) output, and one test window
giving finite numbers. The version must be higher than the model bundled in
the APK and higher than whatever is already published.

Needs a secret key with Storage write access in the environment. Never
commit it, and never put it in the app; phones only read the public bucket.

    export SUPABASE_SECRET_KEY=...   # Project Settings > API Keys > Secret keys
    python scripts/publish_model.py path/to/model.onnx --version 3
    python scripts/publish_model.py path/to/model.onnx --version 3 --dry-run   # check only

To roll back, publish the older model again under a new, higher version.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
SUPABASE_URL = "https://iekkwudnpvijroqvknwq.supabase.co"
BUCKET = "models"
UPDATER_KT = ROOT / "app/app/src/main/java/com/sih26168/deadreckoning/ModelUpdater.kt"
INPUT_SHAPE = [1, 50, 6]
OUTPUT_SHAPE = [1, 50, 2]


def bundled_version() -> int:
    m = re.search(r"BUNDLED_MODEL_VERSION\s*=\s*(\d+)", UPDATER_KT.read_text())
    if not m:
        sys.exit(f"could not find BUNDLED_MODEL_VERSION in {UPDATER_KT}")
    return int(m.group(1))


def check_model(path: Path) -> None:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ins, outs = sess.get_inputs(), sess.get_outputs()
    shapes = ([i.shape for i in ins], [o.shape for o in outs])
    if shapes != ([INPUT_SHAPE], [OUTPUT_SHAPE]):
        sys.exit(f"model shapes {shapes[0]} -> {shapes[1]} don't match {INPUT_SHAPE} -> {OUTPUT_SHAPE}; "
                 "the app would reject it")
    out = sess.run(None, {ins[0].name: np.zeros(INPUT_SHAPE, dtype=np.float32)})[0]
    if out.shape != tuple(OUTPUT_SHAPE) or not np.isfinite(out).all():
        sys.exit("test inference gave a wrong-shaped or non-finite output")


def published_version(dry_run: bool) -> int:
    url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/manifest.json"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return int(json.load(r)["version"])
    except urllib.error.HTTPError as e:
        if e.code in (400, 404):  # bucket or manifest not there yet
            return 0
        raise
    except urllib.error.URLError as e:
        # A paused Supabase project stops resolving in DNS altogether.
        msg = f"can't reach {SUPABASE_URL} ({e.reason}); is the Supabase project paused?"
        if dry_run:
            print(f"warning: {msg} Skipping the already-published version check.")
            return 0
        sys.exit(msg)


def request(method: str, path: str, key: str, body: bytes, content_type: str, extra=None):
    headers = {"apikey": key, "Content-Type": content_type}
    if key.startswith("eyJ"):  # legacy JWT service_role key
        headers["Authorization"] = f"Bearer {key}"
    headers.update(extra or {})
    req = urllib.request.Request(f"{SUPABASE_URL}/storage/v1/{path}", data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--dry-run", action="store_true", help="check the model and print the manifest, upload nothing")
    args = p.parse_args()

    check_model(args.model)
    floor = max(bundled_version(), published_version(args.dry_run))
    if args.version <= floor:
        sys.exit(f"--version must be higher than {floor} (bundled or already published), got {args.version}")

    data = args.model.read_bytes()
    name = f"dead_reckoning_model_v{args.version}.onnx"
    manifest = {"version": args.version, "file": name,
                "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        print("dry run: model checks passed, nothing uploaded")
        return

    key = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        sys.exit("set SUPABASE_SECRET_KEY (or SUPABASE_SERVICE_ROLE_KEY) first")

    status, body = request("POST", "bucket", key, json.dumps({"id": BUCKET, "name": BUCKET, "public": True}).encode(),
                           "application/json")
    if status not in (200, 201) and b"already exists" not in body.lower():
        sys.exit(f"creating bucket failed: HTTP {status} {body[:300]!r}")

    # Versioned model files never change, so they can be cached; the
    # manifest is what moves, so keep its cache short.
    status, body = request("POST", f"object/{BUCKET}/{name}", key, data, "application/octet-stream",
                           {"x-upsert": "true", "cache-control": "max-age=31536000"})
    if status not in (200, 201):
        sys.exit(f"uploading {name} failed: HTTP {status} {body[:300]!r}")
    status, body = request("POST", f"object/{BUCKET}/manifest.json", key, json.dumps(manifest).encode(),
                           "application/json", {"x-upsert": "true", "cache-control": "max-age=60"})
    if status not in (200, 201):
        sys.exit(f"uploading manifest.json failed: HTTP {status} {body[:300]!r}")
    print(f"published v{args.version}: phones fetch it on their next app start and use it on the one after")


if __name__ == "__main__":
    main()
