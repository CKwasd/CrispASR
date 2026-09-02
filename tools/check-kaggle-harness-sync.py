#!/usr/bin/env python3
"""Assert every bundled kaggle_harness.py matches the canonical one.

Each tools/kaggle/<kernel>/ directory ships its own copy of
tools/kaggle/kaggle_harness.py, intended as the fallback used when the
in-kernel `git clone` fails (CPU workers get no internet at all).

⚠ WHETHER THAT FALLBACK CAN FIRE AT ALL IS OPEN — do not assume it protects
you. `kaggle kernels pull` of a pushed script kernel returns exactly one .py
and nothing else (observed 2026-09-03), which is suggestive but NOT conclusive:
pull is selective by design — its `--metadata` flag *generates*
kernel-metadata.json rather than fetching an uploaded one — so that observation
distinguishes nothing about what `push` actually uploads. A runtime probe now
rides on the sidon-quant-cuda kernel (it lists its own directory before the
sm_60 early-exit, so any draw answers it); look for the `upload_probe` step.
The fallback can fire iff `kaggle_harness.py` appears in that listing.

Until the probe reports: keeping the copies byte-identical is cheap hygiene,
but do NOT treat a green run of this check as evidence that the no-internet
path works — that has never been demonstrated either way. If the probe shows
the bundle does not ship, making the fallback real needs a delivery route that
survives no-internet (publishing the harness as a Kaggle DATASET listed in each
kernel's `dataset_sources`, the mechanism the hf-token dataset already uses),
after which the copies and this check could go away — a policy change across
kernels owned by several sessions, so it belongs to the maintainer, not to
whoever reads this next.

Note also that 7 of the 61 bundled copies are untracked (a `.gitignore` glob
added after 54 had already been committed), so they cannot propagate through a
commit at all — harmless while the copies are local-testing conveniences,
load-bearing if anyone revives the fallback without fixing the delivery route.

Nothing kept those copies in sync. On 2026-07-20 there were **four** distinct
versions across 53 files, and the canonical one was used by exactly one kernel.
That is a latent, silent failure: the drift only bites on the clone-failure
path, which is precisely the path nobody exercises until a worker has no
network — and then the kernel runs an arbitrarily old harness (missing token
mount paths, missing ccache fixes) and fails in a way that looks like a Kaggle
problem rather than a stale bundle.

Fix drift with:
    for f in $(find tools/kaggle -name kaggle_harness.py \\
                 -not -path tools/kaggle/kaggle_harness.py); do
        cp tools/kaggle/kaggle_harness.py "$f"
    done
"""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANON = ROOT / "tools" / "kaggle" / "kaggle_harness.py"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if not CANON.is_file():
        print(f"ERROR: canonical harness missing at {CANON}", file=sys.stderr)
        return 1
    want = sha(CANON)
    copies = sorted(p for p in (ROOT / "tools" / "kaggle").rglob("kaggle_harness.py")
                    if p.resolve() != CANON.resolve())
    drifted = [p for p in copies if sha(p) != want]

    print(f"canonical: {CANON.relative_to(ROOT)}  sha256={want[:16]}")
    print(f"bundled copies: {len(copies)}   drifted: {len(drifted)}")
    if drifted:
        print("\nERROR: bundled kaggle_harness.py copies differ from the canonical one:",
              file=sys.stderr)
        for p in drifted:
            print(f"  {p.relative_to(ROOT)}  sha256={sha(p)[:16]}", file=sys.stderr)
        print("\nRe-copy tools/kaggle/kaggle_harness.py over each (see this file's "
              "docstring).", file=sys.stderr)
        return 1
    print("OK: every bundled harness matches the canonical copy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
