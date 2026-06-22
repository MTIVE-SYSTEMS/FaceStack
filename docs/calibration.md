# Threshold calibration

`match_threshold` (default `0.40`) is the cosine-similarity cutoff for "same
person". The default is a literature value; calibrate it on data that resembles
your deployment.

## How the decision works

```
similarity ≥ threshold  → matched (person_id)
similarity < threshold  → unknown
```

The trade-off is two-sided:

| Threshold too **low** | Threshold too **high** |
|---|---|
| strangers accepted (false accept ↑) | real people rejected (false reject ↑) |
| security risk | annoying, misses people |

Pick by use case: access control → high threshold (minimise false accepts);
attendance/counting → lower threshold (don't miss anyone).

## Running it

Lay out a labelled folder (one subfolder per identity):

```
dataset/
  ahmet/   1.jpg 2.jpg ...
  aras/    1.jpg ...
  yigithan/ ...
```

```bash
LD_LIBRARY_PATH=$HOME/rocm-compat python scripts/calibrate.py dataset/
```

It embeds each image, builds same-person and different-person pairs, and reports:

- **similarity distributions** (same vs different) — the real signal; they should
  be cleanly separated;
- **operating points** — best-accuracy, EER (balanced), and threshold at a target FAR;
- **worst pairs by filename** — low same-person sims and high different-person sims,
  to spot bad/mislabelled images;
- a **recommended threshold**.

Apply via `FACESTACK_MATCH_THRESHOLD=0.xx` (env / `.env`).

## Capture tips

- **Variety beats volume.** Per person: different angles, distances, lighting,
  glasses on/off, expressions. Near-identical frontal shots don't stress the threshold.
- **One person per image.** A multi-face photo makes the detector embed the
  *wrong* face — it then matches nothing. The script flags `>1 face` images and
  names the offending pairs; remove or crop those.

## What a clean result looks like

From the 3-person motis calibration (after removing one bad 2-face photo):

```
same-person      : mean 0.690  min 0.492  max 0.961
different-person : mean 0.189  max 0.381
best-accuracy    : thr=0.385  acc=1.000  TAR=1.000  FAR=0.000
```

The distributions are **fully separated** (gap 0.381–0.492), so any threshold in
that gap is perfect on this data — which is why `0.40` is a safe default.

## Caveats

- A 2–3 person set is a **sanity check**, not a precise FAR (the finest
  measurable FAR is `1 / number-of-different-pairs`). For a rigorous baseline use
  a large public set (e.g. LFW) in the same folder layout — the script handles both.
- Clean capture conditions overstate separation. Real CCTV (distance, angle,
  low-res, motion blur) narrows the gap, so **re-calibrate with real footage**
  once deployed, and consider a slightly lower threshold then.
