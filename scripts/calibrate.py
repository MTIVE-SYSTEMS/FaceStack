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
    """Return {person: [(path, embedding)]} and a count of images with no face."""
    embeds: dict[str, list[tuple[str, np.ndarray]]] = {}
    skipped = 0
    for person, paths in people.items():
        items = []
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
            if len(faces) > 1:
                print(f"  ~ {len(faces)} faces in {p}, kept highest-confidence", file=sys.stderr)
            items.append((p, best.embedding))
        if items:
            embeds[person] = items
    return embeds, skipped


def _label(person: str, path: str) -> str:
    return f"{person}/{os.path.basename(path)}"


def recommend_floor(diff: np.ndarray) -> float:
    """A same-person pair below this overlaps impostor territory => suspicious."""
    return float(np.percentile(diff, 95)) if diff.size else 0.0


def make_pairs(embeds: dict[str, list[tuple[str, np.ndarray]]], max_diff: int, seed: int = 0):
    """Return (sims, labels) for same- and different-person pairs."""
    same_s, same_l = [], []
    for person, items in embeds.items():
        for (pa, a), (pb, b) in itertools.combinations(items, 2):
            same_s.append(float(np.dot(a, b)))
            same_l.append((_label(person, pa), _label(person, pb)))

    people = list(embeds.keys())
    rng = np.random.default_rng(seed)
    all_diff = []
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            for pa, a in embeds[people[i]]:
                for pb, b in embeds[people[j]]:
                    all_diff.append((float(np.dot(a, b)), (_label(people[i], pa), _label(people[j], pb))))
    if len(all_diff) > max_diff:
        idx = rng.choice(len(all_diff), size=max_diff, replace=False)
        all_diff = [all_diff[k] for k in idx]
    diff_s = [s for s, _ in all_diff]
    diff_l = [lbl for _, lbl in all_diff]
    return np.array(same_s), same_l, np.array(diff_s), diff_l


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

    same, same_l, diff, diff_l = make_pairs(embeds, args.max_diff_pairs)
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

    # worst cases help spot bad / mislabelled images — name the actual files
    print("\nWorst same-person pairs (should be HIGH — low means a bad/mislabelled image):")
    for k in np.argsort(same)[:5]:
        a, b = same_l[k]
        flag = "  <-- suspicious" if same[k] < recommend_floor(diff) else ""
        print(f"  sim={same[k]:.3f}  {a}  vs  {b}{flag}")
    print("\nWorst different-person pairs (should be LOW — high means look-alikes):")
    for k in np.argsort(diff)[::-1][:5]:
        a, b = diff_l[k]
        print(f"  sim={diff[k]:.3f}  {a}  vs  {b}")

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
