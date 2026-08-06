"""Fetch the pinned Metamath databases the evaluator verifies proofs against.

`run_eval.py` reports Metamath validity only when `--mm-dir` supplies set.mm,
iset.mm and nf.mm whose SHA-256 digests match `corpus-v3/metamath_sources.json`.
Without them `metamath_runtime_availability` degrades to "unavailable" silently,
so the run completes and quietly omits an endpoint. This fetches them at the
pinned commit and refuses anything that does not match.

    python fetch_metamath_sources.py --manifest corpus-v3/corpus-v3/metamath_sources.json --out mm
"""

import argparse
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/metamath/set.mm/{commit}/{filename}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ssl_context(ca_bundle: Path | None) -> ssl.SSLContext | None:
    """Trust an explicit CA bundle when a proxy re-signs TLS.

    Only needed behind TLS interception; on a clean host the default chain
    works and this returns None. Verification stays on either way -- the
    downloads are digest-checked afterwards regardless.
    """
    candidate = ca_bundle or (
        Path(os.environ["SSL_CERT_FILE"]) if os.environ.get("SSL_CERT_FILE") else None
    )
    if candidate is None:
        return None
    if not candidate.is_file():
        raise SystemExit(f"CA bundle does not exist: {candidate}")
    return ssl.create_default_context(cafile=str(candidate))


def fetch(url: str, target: Path, context: ssl.SSLContext | None) -> None:
    try:
        with urllib.request.urlopen(url, context=context) as response:
            if response.status != 200:
                raise RuntimeError(f"{url} returned HTTP {response.status}")
            target.write_bytes(response.read())
        return
    except (ssl.SSLError, urllib.error.URLError) as error:
        # urlopen wraps the SSLError, so unwrap before deciding this is a TLS
        # problem rather than a genuine network or HTTP failure.
        reason = getattr(error, "reason", error)
        if not isinstance(reason, ssl.SSLError):
            raise

    # Behind TLS interception, OpenSSL may reject a re-signed chain that the OS
    # store accepts. curl uses the platform store, so it gets through where
    # Python cannot. The digest check afterwards is what actually guarantees
    # integrity, so falling back costs no safety.
    curl = shutil.which("curl")
    if curl is None:
        raise
    result = subprocess.run(
        [curl, "-sSL", "--fail", "-o", str(target), url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"urllib rejected the TLS chain and curl exited {result.returncode}: "
            f"{result.stderr.strip()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if a valid copy exists"
    )
    parser.add_argument(
        "--ca-bundle",
        type=Path,
        default=None,
        help="CA bundle to trust; also read from SSL_CERT_FILE. Needed only behind "
        "TLS interception",
    )
    args = parser.parse_args()

    context = ssl_context(args.ca_bundle)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    commit = manifest["commit"]
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"repository {manifest['repository']}")
    print(f"commit     {commit}\n")

    failures = []
    for filename, entry in sorted(manifest["files"].items()):
        expected = entry["sha256"]
        target = args.out / filename

        if target.exists() and not args.force:
            actual = sha256_file(target)
            if actual == expected:
                size = target.stat().st_size / 1024 / 1024
                print(f"  {filename:<10} cached, verified   {size:>7.1f} MB")
                continue
            print(f"  {filename:<10} cached copy is stale, re-downloading")

        url = RAW.format(commit=commit, filename=filename)
        print(f"  {filename:<10} downloading ...", end="", flush=True)
        try:
            fetch(url, target, context)
        except Exception as error:
            print(f" FAILED: {error}")
            failures.append(filename)
            continue

        actual = sha256_file(target)
        size = target.stat().st_size / 1024 / 1024
        if actual != expected:
            print(f" DIGEST MISMATCH  {size:>7.1f} MB")
            print(f"      expected {expected}")
            print(f"      got      {actual}")
            failures.append(filename)
            continue
        print(f" verified   {size:>7.1f} MB")

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1

    print(f"\nAll {len(manifest['files'])} databases verified against the corpus snapshot.")
    print(f"Pass --mm-dir {args.out.resolve()} to run_eval.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
