"""
Publishes an ONNX model so the Android app picks it up without a new APK
(see app/.../ModelUpdater.kt).

The app runs a two-model ensemble (see src/export_ensemble.py); each model
is published on its own with --role:

    slow    model_slow.onnx, exactly one (1, 50, 6) input
    fast    model_fast.onnx, exactly one (1, 50, 7) input (6 IMU + speed)
    legacy  the single (1, 50, 6) model app versions before the ensemble
            read from the manifest's top-level fields

manifest.json keeps one entry per role under "models", each with its own
version, file, sha256 and size_bytes; publishing one role rewrites only that
entry and leaves the others (and the legacy top-level fields) as they were.

Uploads the model, then manifest.json, to the public "models" bucket in the
project's Supabase Storage (creating the bucket the first time). The model
goes up first, so a phone never sees a manifest pointing at a missing file.
Before uploading, the model is checked the same way the app checks it: the
role's exact input shape, one (1, 50, 2) output, and one test window giving
finite numbers. The version must be higher than the role's model bundled in
the APK and higher than whatever is already published for that role.

Needs a secret key with Storage write access in the environment. Never
commit it, and never put it in the app; phones only read the public bucket.

    export SUPABASE_SECRET_KEY=...   # Project Settings > API Keys > Secret keys
    python scripts/publish_model.py checkpoints/model_fast.onnx --role fast --version 2
    python scripts/publish_model.py <model.onnx> --role slow --version 2 --dry-run   # check only

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
CHANNELS = {"slow": 6, "fast": 7, "legacy": 6}
OUTPUT_SHAPE = [1, 50, 2]
# The last single model bundled before the ensemble (best_v3chain.pt).
LEGACY_BUNDLED_VERSION = 2


def input_shape(role: str) -> list:
    return [1, 50, CHANNELS[role]]


def bundled_version(role: str) -> int:
    if role == "legacy":
        return LEGACY_BUNDLED_VERSION
    const = f"BUNDLED_{role.upper()}_VERSION"
    m = re.search(rf"{const}\s*=\s*(\d+)", UPDATER_KT.read_text())
    if not m:
        sys.exit(f"could not find {const} in {UPDATER_KT}")
    return int(m.group(1))


def check_model(path: Path, role: str) -> None:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ins, outs = sess.get_inputs(), sess.get_outputs()
    shapes = ([i.shape for i in ins], [o.shape for o in outs])
    want = input_shape(role)
    if shapes != ([want], [OUTPUT_SHAPE]):
        sys.exit(f"model shapes {shapes[0]} -> {shapes[1]} don't match {want} -> {OUTPUT_SHAPE} "
                 f"for role {role}; the app would reject it")
    out = sess.run(None, {ins[0].name: np.zeros(want, dtype=np.float32)})[0]
    if out.shape != tuple(OUTPUT_SHAPE) or not np.isfinite(out).all():
        sys.exit("test inference gave a wrong-shaped or non-finite output")


def role_version(manifest: dict, role: str) -> int:
    entry = manifest if role == "legacy" else manifest.get("models", {}).get(role, {})
    return int(entry.get("version", 0))


def updated_manifest(current: dict, role: str, entry: dict) -> dict:
    """current with only this role's entry replaced."""
    out = json.loads(json.dumps(current))
    if role == "legacy":
        out.update(entry)
    else:
        out.setdefault("models", {})[role] = entry
    return out


def published_manifest(dry_run: bool) -> dict:
    url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/manifest.json"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (400, 404):  # bucket or manifest not there yet
            return {}
        raise
    except urllib.error.URLError as e:
        # A paused Supabase project stops resolving in DNS altogether.
        msg = f"can't reach {SUPABASE_URL} ({e.reason}); is the Supabase project paused?"
        if dry_run:
            print(f"warning: {msg} Skipping the already-published version check.")
            return {}
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
    p.add_argument("--role", choices=sorted(CHANNELS), required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--dry-run", action="store_true", help="check the model and print the manifest, upload nothing")
    args = p.parse_args()

    check_model(args.model, args.role)
    current = published_manifest(args.dry_run)
    floor = max(bundled_version(args.role), role_version(current, args.role))
    if args.version <= floor:
        sys.exit(f"--version must be higher than {floor} (bundled or already published {args.role}), "
                 f"got {args.version}")

    data = args.model.read_bytes()
    name = (f"dead_reckoning_model_v{args.version}.onnx" if args.role == "legacy"
            else f"model_{args.role}_v{args.version}.onnx")
    entry = {"version": args.version, "file": name,
             "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    manifest = updated_manifest(current, args.role, entry)
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
    print(f"published {args.role} v{args.version}: phones fetch it on their next app start and use it on the one after")


if __name__ == "__main__":
    main()
