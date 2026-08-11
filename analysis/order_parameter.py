"""
Order parameter sweep for the Boids model.

Measures the polarization (Vicsek) order parameter

    phi = | (1/N) * sum_i  v_i / |v_i| |

as a function of the neighbour radius. phi = 1 when every bird moves in the
same direction, and falls toward ~1/sqrt(N) for uncorrelated headings (0.13
for N = 60), which is the disordered baseline rather than 0.

The simulation is driven headlessly by importing the Boid class from
flight_model, so the dynamics measured here are exactly the dynamics the
renderer displays -- there is no second copy of the rules to drift out of sync.

Usage:
    python analysis/order_parameter.py                 # full sweep
    python analysis/order_parameter.py --quick         # fast smoke test
    python analysis/order_parameter.py --knob alignment_radius
"""

import argparse
import csv
import math
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Neighbour-radius values to sweep, per knob.
#   visual_range     -- RADIUS_MULTIPLIER, the "Visual Range" slider in the UI.
#                       Scales separation, alignment and cohesion radii together.
#   alignment_radius -- ALIGNMENT_RADIUS only, holding separation fixed, which
#                       isolates the alignment interaction from crowding effects.
SWEEPS = {
    "visual_range": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0],
    "alignment_radius": [6, 12, 24, 36, 48, 72, 96, 144, 192, 288, 384],
}


def polarization(boids):
    """Normalised mean heading. 1.0 = perfectly aligned flock."""
    v = np.array([(b.vel.x, b.vel.y, b.vel.z) for b in boids], dtype=float)
    speeds = np.linalg.norm(v, axis=1)
    moving = speeds > 1e-9
    if not moving.any():
        return 0.0
    unit = v[moving] / speeds[moving, None]
    return float(np.linalg.norm(unit.mean(axis=0)))


def mean_neighbour_count(boids, fm):
    """Average number of boids inside the alignment radius."""
    p = np.array([(b.pos.x, b.pos.y, b.pos.z) for b in boids], dtype=float)
    d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
    r = (fm.ALIGNMENT_RADIUS * fm.RADIUS_MULTIPLIER) ** 2
    within = (d2 <= r).sum(axis=1) - 1  # exclude self
    return float(within.mean())


def rms_spread(boids):
    """RMS distance of birds from the flock centroid -- how condensed the flock is."""
    p = np.array([(b.pos.x, b.pos.y, b.pos.z) for b in boids], dtype=float)
    return float(np.sqrt(((p - p.mean(axis=0)) ** 2).sum(axis=1).mean()))


def run_single(args):
    """One independent realisation. Returns (value, seed, mean_phi, sem_phi, mean_k, series)."""
    knob, value, seed, n_boids, burn_in, measure, keep_series = args

    import flight_model as fm

    if knob == "visual_range":
        fm.RADIUS_MULTIPLIER = value
    elif knob == "alignment_radius":
        fm.ALIGNMENT_RADIUS = value
    else:
        raise ValueError(f"unknown knob: {knob}")

    random.seed(seed)
    boids = [
        fm.Boid(random.uniform(0, fm.DEPTH), random.uniform(0, fm.DEPTH), (255, 255, 255))
        for _ in range(n_boids)
    ]

    phis = []
    kcounts = []
    spreads = []
    series = []

    for t in range(burn_in + measure):
        for b in boids:
            s = b.separation(boids)
            a = b.alignment(boids)
            c = b.cohesion(boids)
            b.apply_force(s * fm.SEPARATION_WEIGHT)
            b.apply_force(a * fm.ALIGNMENT_WEIGHT)
            b.apply_force(c * fm.COHESION_WEIGHT)
        for b in boids:
            b.update()

        if keep_series and t % 20 == 0:
            series.append((t, polarization(boids)))

        if t >= burn_in:
            phis.append(polarization(boids))
            if (t - burn_in) % 25 == 0:
                kcounts.append(mean_neighbour_count(boids, fm))
                spreads.append(rms_spread(boids))

    phis = np.array(phis)
    return (
        value,
        seed,
        float(phis.mean()),
        float(phis.std(ddof=1) / math.sqrt(len(phis))),
        float(np.mean(kcounts)) if kcounts else float("nan"),
        series,
        float(np.mean(spreads)) if spreads else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", default="visual_range", choices=list(SWEEPS))
    ap.add_argument("--boids", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=12)
    # The flock condenses slowly from a sparse random start; polarization is only
    # meaningful once the aggregation transient has passed (~3-5k steps at the
    # shipped default radius), so the burn-in is deliberately long.
    ap.add_argument("--burn-in", type=int, default=6000)
    ap.add_argument("--measure", type=int, default=3000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--values", default="",
                    help="comma-separated radii, overriding the default sweep")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames, so a targeted rerun "
                         "does not overwrite the main sweep")
    args = ap.parse_args()

    if args.quick:
        args.seeds, args.burn_in, args.measure = 2, 200, 200

    values = ([float(v) for v in args.values.split(",")] if args.values
              else SWEEPS[args.knob])
    jobs = [
        (args.knob, v, 1000 + s, args.boids, args.burn_in, args.measure, s == 0)
        for v in values
        for s in range(args.seeds)
    ]

    procs = args.procs or min(os.cpu_count() or 1, 8)
    print(
        f"knob={args.knob}  values={len(values)}  seeds={args.seeds}  "
        f"runs={len(jobs)}  steps/run={args.burn_in + args.measure}  procs={procs}",
        flush=True,
    )

    with Pool(procs) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_single, jobs), 1):
            results.append(r)
            print(f"  [{i}/{len(jobs)}] {args.knob}={r[0]:g} seed={r[1]} phi={r[2]:.3f}", flush=True)

    tag = args.knob + (f"_{args.tag}" if args.tag else "")
    raw_path = os.path.join(OUT_DIR, f"order_parameter_{tag}_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([args.knob, "seed", "mean_phi", "sem_phi_within_run",
                    "mean_neighbours", "rms_spread"])
        for value, seed, mphi, sem, k, _, sp in sorted(results, key=lambda r: (r[0], r[1])):
            w.writerow([value, seed, f"{mphi:.6f}", f"{sem:.6f}", f"{k:.4f}", f"{sp:.2f}"])

    summary = []
    for v in values:
        rows = [r for r in results if r[0] == v]
        phis = np.array([r[2] for r in rows])
        ks = np.array([r[4] for r in rows])
        sps = np.array([r[6] for r in rows])
        summary.append(
            (
                v,
                phis.mean(),
                phis.std(ddof=1) / math.sqrt(len(phis)) if len(phis) > 1 else 0.0,
                np.nanmean(ks),
                np.nanmean(sps),
            )
        )

    sum_path = os.path.join(OUT_DIR, f"order_parameter_{tag}.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([args.knob, "phi_mean", "phi_sem_across_seeds",
                    "mean_neighbours", "rms_spread"])
        for v, m, sem, k, sp in summary:
            w.writerow([v, f"{m:.6f}", f"{sem:.6f}", f"{k:.4f}", f"{sp:.2f}"])

    baseline = 1.0 / math.sqrt(args.boids)
    print(f"\n  {args.knob:>16}  {'phi':>7} {'SEM':>7} {'<k>':>7} {'spread':>8}")
    for v, m, sem, k, sp in summary:
        print(f"  {v:>16g}  {m:7.3f} {sem:7.3f} {k:7.2f} {sp:8.1f}")
    print(f"\n  disordered baseline 1/sqrt(N) = {baseline:.3f}")
    print(f"  wrote {raw_path}\n  wrote {sum_path}")

    if not args.tag:
        make_plot(args, values, summary, results, baseline, tag)


def make_plot(args, values, summary, results, baseline, tag):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.3))

    xs = [s[0] for s in summary]
    ys = [s[1] for s in summary]
    es = [s[2] for s in summary]

    ax1.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.6, color="#2b6cb0")
    ax1.axhline(baseline, ls="--", lw=1.2, color="#a0aec0",
                label=f"disordered baseline $1/\\sqrt{{N}}$ = {baseline:.2f}")
    ax1.set_xscale("log")
    ax1.set_xlabel({"visual_range": "Visual range (radius multiplier)",
                    "alignment_radius": "Alignment radius"}[args.knob])
    ax1.set_ylabel(r"Order parameter  $\phi$")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"Flock polarization vs neighbour radius\n(N={args.boids}, {args.seeds} seeds, "
                  f"{args.measure} steps after {args.burn_in}-step burn-in)", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    for v in values:
        rows = [r for r in results if r[0] == v and r[5]]
        for r in rows:
            t = [p[0] for p in r[5]]
            y = [p[1] for p in r[5]]
            ax2.plot(t, y, lw=1.0, alpha=0.85, label=f"{v:g}")
    ax2.axvline(args.burn_in, ls=":", color="#718096", lw=1.2)
    ax2.text(args.burn_in, 1.02, " burn-in ends", fontsize=7, color="#718096")
    ax2.set_xlabel("Simulation step")
    ax2.set_ylabel(r"$\phi(t)$")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("Convergence to steady state (one seed per radius)", fontsize=9)
    ax2.legend(fontsize=6, ncol=2, title=args.knob, title_fontsize=6)
    ax2.grid(alpha=0.25)

    # Mechanism: order rises because cohesion condenses the flock, which raises
    # the neighbour count, which lets alignment propagate across the group.
    spreads = [s[4] for s in summary]
    ks = [s[3] for s in summary]
    ax3.plot(xs, spreads, marker="s", lw=1.6, color="#c05621", label="RMS spread from centroid")
    ax3.set_xscale("log")
    ax3.set_xlabel(ax1.get_xlabel())
    ax3.set_ylabel("RMS spread (world units)", color="#c05621")
    ax3.tick_params(axis="y", labelcolor="#c05621")
    ax3.grid(alpha=0.25)
    ax3b = ax3.twinx()
    ax3b.plot(xs, ks, marker="^", lw=1.6, ls="--", color="#2f855a", label="mean neighbours")
    ax3b.set_ylabel(r"Mean neighbours $\langle k \rangle$", color="#2f855a")
    ax3b.tick_params(axis="y", labelcolor="#2f855a")
    ax3b.axhline(args.boids - 1, ls=":", lw=1.0, color="#a0aec0")
    ax3b.text(xs[0], args.boids - 1, " all-to-all (N-1)", fontsize=7, color="#718096", va="bottom")
    ax3.set_title("Mechanism: cohesion condenses the flock,\nraising connectivity", fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"order_parameter_{tag}.png")
    fig.savefig(path, dpi=160)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
