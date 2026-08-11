"""
Render the order-parameter figure from results already on disk.

Kept separate from order_parameter.py so the figure can be restyled without
repeating a ~20 minute sweep. Panels 1 and 3 read the summary CSV. Panel 2
needs phi(t) traces, which are simulated once for a small subset of radii and
then cached to convergence_<knob>.csv.

Usage:
    python analysis/make_figure.py --knob visual_range
    python analysis/make_figure.py --knob visual_range --refresh-traces
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# A readable handful of radii spanning disordered -> transition -> saturated.
TRACE_SUBSET = {
    "visual_range": [0.5, 1.5, 3.0, 5.0, 16.0],
    "alignment_radius": [12, 48, 96, 192, 384],
}

XLABEL = {
    "visual_range": "Visual range (radius multiplier)",
    "alignment_radius": "Alignment radius",
}


def simulate_trace(args):
    """phi(t) for one radius, sampled every `every` steps."""
    knob, value, seed, n_boids, steps, every = args
    from order_parameter import polarization
    import flight_model as fm

    if knob == "visual_range":
        fm.RADIUS_MULTIPLIER = value
    else:
        fm.ALIGNMENT_RADIUS = value

    random.seed(seed)
    boids = [
        fm.Boid(random.uniform(0, fm.DEPTH), random.uniform(0, fm.DEPTH), (255, 255, 255))
        for _ in range(n_boids)
    ]
    out = []
    for t in range(steps):
        for b in boids:
            b.apply_force(b.separation(boids) * fm.SEPARATION_WEIGHT)
            b.apply_force(b.alignment(boids) * fm.ALIGNMENT_WEIGHT)
            b.apply_force(b.cohesion(boids) * fm.COHESION_WEIGHT)
        for b in boids:
            b.update()
        if t % every == 0:
            out.append((t, polarization(boids)))
    return value, out


def load_or_build_traces(knob, n_boids, steps, every, refresh):
    path = os.path.join(OUT_DIR, f"convergence_{knob}.csv")
    if os.path.exists(path) and not refresh:
        traces = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                traces.setdefault(float(row["value"]), []).append(
                    (int(row["step"]), float(row["phi"]))
                )
        return traces

    values = TRACE_SUBSET[knob]
    jobs = [(knob, v, 1000, n_boids, steps, every) for v in values]
    print(f"simulating {len(jobs)} convergence traces ({steps} steps each)...", flush=True)
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        results = pool.map(simulate_trace, jobs)

    traces = {v: s for v, s in results}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["value", "step", "phi"])
        for v in values:
            for t, p in traces[v]:
                w.writerow([v, t, f"{p:.6f}"])
    print(f"  wrote {path}", flush=True)
    return traces


def smooth(y, window):
    if len(y) < window or window < 2:
        return np.asarray(y, dtype=float)
    k = np.ones(window) / window
    return np.convolve(np.asarray(y, dtype=float), k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", default="visual_range", choices=list(TRACE_SUBSET))
    ap.add_argument("--boids", type=int, default=60)
    ap.add_argument("--burn-in", type=int, default=6000)
    ap.add_argument("--trace-steps", type=int, default=9000)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--smooth-window", type=int, default=15)
    ap.add_argument("--refresh-traces", action="store_true")
    args = ap.parse_args()

    summary_path = os.path.join(OUT_DIR, f"order_parameter_{args.knob}.csv")
    if not os.path.exists(summary_path):
        sys.exit(f"missing {summary_path} -- run order_parameter.py --knob {args.knob} first")

    xs, phis, sems, ks, spreads = [], [], [], [], []
    with open(summary_path) as f:
        for row in csv.DictReader(f):
            xs.append(float(row[args.knob]))
            phis.append(float(row["phi_mean"]))
            sems.append(float(row["phi_sem_across_seeds"]))
            ks.append(float(row["mean_neighbours"]))
            spreads.append(float(row["rms_spread"]))

    traces = load_or_build_traces(
        args.knob, args.boids, args.trace_steps, args.every, args.refresh_traces
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = 1.0 / math.sqrt(args.boids)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # --- Panel 1: order parameter vs radius -------------------------------
    ax1.errorbar(xs, phis, yerr=sems, marker="o", capsize=3, lw=1.8,
                 color="#2b6cb0", zorder=3)
    ax1.axhline(baseline, ls="--", lw=1.2, color="#a0aec0",
                label=f"disordered baseline $1/\\sqrt{{N}}$ = {baseline:.2f}")

    # Half-way crossing between disorder and saturation.
    sat = max(phis)
    half = (baseline + sat) / 2.0
    xc = None
    for i in range(1, len(xs)):
        if (phis[i - 1] - half) * (phis[i] - half) <= 0 and phis[i] != phis[i - 1]:
            lo, hi = math.log10(xs[i - 1]), math.log10(xs[i])
            frac = (half - phis[i - 1]) / (phis[i] - phis[i - 1])
            xc = 10 ** (lo + frac * (hi - lo))
            break
    # Only call the crossing a disorder->order *threshold* when the low end of the
    # sweep actually sits at the disordered baseline. For the alignment-radius
    # sweep it does not (cohesion still orders the flock on its own), so the
    # crossing is only a half-max point and must not be labelled a threshold.
    spans_disorder = min(phis) < baseline * 1.5
    if xc:
        label = "ordering threshold" if spans_disorder else "half-max"
        ax1.axvline(xc, ls=":", lw=1.3, color="#dd6b20")
        ax1.annotate(f"{label} ≈ {xc:.2g}", xy=(xc, half),
                     xytext=(xc * 1.15, half - 0.17), fontsize=8, color="#dd6b20",
                     arrowprops=dict(arrowstyle="->", color="#dd6b20", lw=1))
    if not spans_disorder:
        ax1.text(0.02, 0.02,
                 "low end stays ordered: cohesion alone\ncondenses and polarizes the flock",
                 transform=ax1.transAxes, fontsize=7.5, color="#4a5568", va="bottom")

    ax1.set_xscale("log")
    ax1.set_xlabel(XLABEL[args.knob])
    ax1.set_ylabel(r"Order parameter  $\phi$")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"Flock polarization vs neighbour radius\n"
                  f"(N={args.boids}, 12 seeds, 3000 steps after {args.burn_in}-step burn-in)",
                  fontsize=9)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)

    # --- Panel 2: convergence ---------------------------------------------
    cmap = plt.get_cmap("viridis")
    subset = TRACE_SUBSET[args.knob]
    for i, v in enumerate(subset):
        ser = traces.get(v) or traces.get(float(v))
        if not ser:
            continue
        ser = sorted(ser)
        t = np.array([p[0] for p in ser], dtype=float)
        y = np.array([p[1] for p in ser], dtype=float)
        ys = smooth(y, args.smooth_window)
        ts = t[: len(ys)] + (args.smooth_window // 2) * args.every
        ax2.plot(ts, ys, lw=1.7, color=cmap(i / max(1, len(subset) - 1)),
                 label=f"{v:g}", zorder=3)
    ax2.axhline(baseline, ls="--", lw=1.1, color="#a0aec0")
    ax2.axvline(args.burn_in, ls=":", lw=1.3, color="#718096")
    ax2.text(args.burn_in, 1.015, " measurement starts", fontsize=7.5, color="#718096")
    ax2.set_xlabel("Simulation step")
    ax2.set_ylabel(r"$\phi(t)$")
    ax2.set_ylim(0, 1.07)
    ax2.set_title(f"Convergence to steady state\n"
                  f"(one seed, rolling mean over {args.smooth_window * args.every} steps)",
                  fontsize=9)
    ax2.legend(fontsize=7.5, title=XLABEL[args.knob].split(" (")[0], title_fontsize=7.5,
               loc="lower right")
    ax2.grid(alpha=0.25)

    # --- Panel 3: mechanism ------------------------------------------------
    ax3.plot(xs, spreads, marker="s", lw=1.8, color="#c05621")
    ax3.set_xscale("log")
    ax3.set_xlabel(XLABEL[args.knob])
    ax3.set_ylabel("RMS spread from centroid", color="#c05621")
    ax3.tick_params(axis="y", labelcolor="#c05621")
    ax3.grid(alpha=0.25)
    ax3b = ax3.twinx()
    ax3b.plot(xs, ks, marker="^", ls="--", lw=1.8, color="#2f855a")
    ax3b.set_ylabel(r"Mean neighbours $\langle k \rangle$", color="#2f855a")
    ax3b.tick_params(axis="y", labelcolor="#2f855a")
    ax3b.axhline(args.boids - 1, ls=":", lw=1.0, color="#a0aec0")
    ax3b.text(xs[0], args.boids - 1, " all-to-all (N-1)", fontsize=7,
              color="#718096", va="bottom")
    if xc:
        ax3.axvline(xc, ls=":", lw=1.3, color="#dd6b20")
    ax3.set_title("Mechanism: cohesion condenses the flock (orange),\n"
                  "raising connectivity (green)", fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"order_parameter_{args.knob}.png")
    fig.savefig(path, dpi=160)
    print(f"  wrote {path}")
    if xc:
        print(f"  ordering threshold ({args.knob} at phi halfway) ~ {xc:.3g}")


if __name__ == "__main__":
    main()
