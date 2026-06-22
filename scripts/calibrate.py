"""Threshold calibration from a labelled face folder.

Layout (one subfolder per identity, any image names):

    dataset/
      alice/  a1.jpg a2.jpg a3.jpg ...
      bob/    b1.jpg b2.jpg ...
      carol/  ...

It embeds each image (largest detected face), builds same-person and
different-person pairs, and reports the similarity distributions plus a
recommended match_threshold. Works for a tiny 2-3 person sanity set today and
for a large public set (e.g. LFW, same layout) later.

    LD_LIBRARY_PATH=$HOME/rocm-compat python scripts/calibrate.py dataset/

Capture tip: variety beats volume — per person shoot different angles,
distances, lighting, glasses on/off. Near-identical frontal shots won't stress
the threshold.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import cv2
import numpy as np

from facestack import Config, FaceEngine

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dataset(root: str) -> dict[str, list[str]]:
    people: dict[str, list[str]] = {}
    for person in sorted(os.listdir(root)):
        pdir = os.path.join(root, person)
        if not os.path.isdir(pdir):
            continue
        imgs = [
            os.path.join(pdir, f)
            for f in sorted(os.listdir(pdir))
            if os.path.splitext(f)[1].lower() in IMG_EXT
        ]
        if imgs:
            people[person] = imgs
    return people


def embed_all(engine: FaceEngine, people: dict[str, list[str]]):
    """Return {person: [embeddings]} and a count of images with no detectable face."""
    embeds: dict[str, list[np.ndarray]] = {}
    skipped = 0
    for person, paths in people.items():
        vecs = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                print(f"  ! unreadable: {p}", file=sys.stderr)
                skipped += 1
                continue
            faces = engine.embed_frame(img)
            if not faces:
                print(f"  ! no face: {p}", file=sys.stderr)
                skipped += 1
                continue
            best = max(faces, key=lambda f: f.det_score)  # one face per image
            vecs.append(best.embedding)
        if vecs:
            embeds[person] = vecs
    return embeds, skipped


def make_pairs(embeds: dict[str, list[np.ndarray]], max_diff: int, seed: int = 0):
    same, diff = [], []
    for vecs in embeds.values():
        for a, b in itertools.combinations(vecs, 2):
            same.append(float(np.dot(a, b)))

    people = list(embeds.keys())
    rng = np.random.default_rng(seed)
    cross = [(i, j) for i in range(len(people)) for j in range(i + 1, len(people))]
    # collect all cross-person image pairs, then sample for speed if huge
    all_diff = []
    for i, j in cross:
        for a in embeds[people[i]]:
            for b in embeds[people[j]]:
                all_diff.append((a, b))
    if len(all_diff) > max_diff:
        idx = rng.choice(len(all_diff), size=max_diff, replace=False)
        all_diff = [all_diff[k] for k in idx]
    diff = [float(np.dot(a, b)) for a, b in all_diff]
    return np.array(same), np.array(diff)


def summarize(name: str, x: np.ndarray) -> None:
    if x.size == 0:
        print(f"  {name}: (none)")
        return
    print(
        f"  {name:18s} n={x.size:5d}  mean={x.mean():.3f}  std={x.std():.3f}  "
        f"min={x.min():.3f}  max={x.max():.3f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dataset root (one subfolder per person)")
    ap.add_argument("--target-far", type=float, default=0.01, help="target false-accept rate")
    ap.add_argument("--max-diff-pairs", type=int, default=20000)
    args = ap.parse_args()

    people = load_dataset(args.root)
    if len(people) < 2:
        print("Need at least 2 people (subfolders) to calibrate.", file=sys.stderr)
        return 2
    print(f"People: {len(people)} -> {', '.join(f'{k}({len(v)})' for k, v in people.items())}")

    engine = FaceEngine(Config())
    print(f"engine: active={engine.active_providers} gpu={engine.on_gpu}\n")

    embeds, skipped = embed_all(engine, people)
    n_imgs = sum(len(v) for v in embeds.values())
    print(f"\nEmbedded {n_imgs} faces ({skipped} images skipped: unreadable / no face).")

    same, diff = make_pairs(embeds, args.max_diff_pairs)
    if same.size == 0:
        print("No same-person pairs — give each person >=2 images.", file=sys.stderr)
        return 2
    print("\nSimilarity distributions:")
    summarize("same-person", same)
    summarize("different-person", diff)

    # finest measurable FAR is 1 / number-of-different-pairs
    far_resolution = 1.0 / max(diff.size, 1)
    print(f"\n(Finest measurable FAR with this data: {far_resolution:.4f} = 1/{diff.size})")

    # threshold sweep
    ts = np.round(np.arange(-0.1, 1.0, 0.005), 3)
    best_acc, best_t = -1.0, 0.40
    eer_t, eer_gap = 0.40, 9.9
    far_t = None
    for t in ts:
        tpr = float((same >= t).mean())  # true accept (same correctly matched)
        far = float((diff >= t).mean())  # false accept (different wrongly matched)
        frr = 1.0 - tpr
        acc = (same.size * tpr + diff.size * (1.0 - far)) / (same.size + diff.size)
        if acc > best_acc:
            best_acc, best_t = acc, t
        if abs(far - frr) < eer_gap:
            eer_gap, eer_t = abs(far - frr), t
        if far <= args.target_far and far_t is None:
            far_t = (t, tpr, far)

    def stat_at(t):
        tpr = float((same >= t).mean())
        far = float((diff >= t).mean())
        return tpr, far

    print("\nOperating points:")
    tpr, far = stat_at(best_t)
    print(f"  best-accuracy : thr={best_t:.3f}  acc={best_acc:.3f}  TAR={tpr:.3f}  FAR={far:.3f}")
    tpr, far = stat_at(eer_t)
    print(f"  balanced (EER): thr={eer_t:.3f}  TAR={tpr:.3f}  FAR={far:.3f}")
    if far_t:
        print(f"  FAR<={args.target_far:.3f}  : thr={far_t[0]:.3f}  TAR={far_t[1]:.3f}  FAR={far_t[2]:.3f}")
    else:
        print(f"  FAR<={args.target_far:.3f}  : not reachable with this little data")

    # worst cases help spot bad enrollment images
    print("\nHardest pairs (inspect these images):")
    if same.size:
        print(f"  lowest same-person sim : {same.min():.3f}  (a true pair that barely matches)")
    if diff.size:
        print(f"  highest different sim  : {diff.max():.3f}  (two different people looking alike)")

    rec = eer_t
    print(f"\nRecommended FACESTACK_MATCH_THRESHOLD = {rec:.2f}   (current default: 0.40)")
    if len(people) < 5 or diff.size < 200:
        print(
            "NOTE: small sample — treat this as a sanity check, not a precise FAR.\n"
            "      The same/different distributions above are the real signal: they\n"
            "      should be cleanly separated. Add more people/conditions to tighten."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
