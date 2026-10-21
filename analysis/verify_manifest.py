"""Re-hash the saved-result inputs and the outputs and compare with outputs/MANIFEST.sha256.

    python -m analysis.verify_manifest [--v2 PATH] [--out PATH]

Exit code 0 if every hash matches; 1 otherwise (each mismatch is printed).
"""
import sys
from analysis import common as C

if __name__ == "__main__":
    C.cli(__doc__)
    p = C.paths(); man = p["out"] / "MANIFEST.sha256"
    bad = 0
    for line in man.read_text().splitlines():
        digest, ref = line.split("  ", 1); root, rel = ref.split(":", 1)
        base = {"outputs": p["out"], "annotation": C.REPO / "annotation"}.get(root) or p[root]
        path = base / rel
        ok = path.exists() and C.sha256(path) == digest
        bad += not ok
        if not ok: print("MISMATCH", ref)
    n = len(man.read_text().splitlines())
    print(f"{n - bad}/{n} files match {man}")
    sys.exit(1 if bad else 0)
