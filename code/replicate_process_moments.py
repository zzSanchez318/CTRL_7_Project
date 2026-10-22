#!/usr/bin/env python3
"""Reproduce the reference-tracking examples in Figures 5 and 6.

Paper:
  D. Vrancic, P. Bistak, M. Huba, and P. Moura Oliveira,
  "A New Closed-Loop Control Paradigm Based on Process Moments",
  Mathematics 2025, 13, 244.

The implementation follows Eqs. (9), (13), (14), and (15):
  Gp(s) = 1.5 / (1 + 2 s + s^2) * exp(-0.4 s)
  y_inf_dot = (KPR * u - y) / t_ar
  u_c = y/KPR + (KTR/KPR) * (r - y_inf)
  u = clip(u_c, u_min, u_max) for the limited case.

Only the reference-tracking experiment in Figures 5 and 6 is reproduced.
Disturbance rejection and model mismatch are outside this target.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cont2discrete


@dataclass(frozen=True)
class PlantConfig:
    kpr: float = 1.5
    t_ar: float = 2.4
    delay: float = 0.4
    step_time: float = 1.0
    dt: float = 0.001
    t_end: float = 10.0


@dataclass
class SimulationResult:
    time: np.ndarray
    reference: np.ndarray
    output: np.ndarray
    input_applied: np.ndarray
    input_unlimited: np.ndarray
    y_inf: np.ndarray
    gain: Optional[float]
    saturation: Optional[tuple[float, float]]
    mode: str


def _discrete_plant(cfg: PlantConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact ZOH discretization of 1.5/(s+1)^2, excluding the pure delay."""
    a = np.array([[0.0, 1.0], [-1.0, -2.0]], dtype=float)
    b = np.array([[0.0], [cfg.kpr]], dtype=float)
    c = np.array([[1.0, 0.0]], dtype=float)
    d = np.zeros((1, 1), dtype=float)
    ad, bd, cd, _, _ = cont2discrete((a, b, c, d), cfg.dt, method="zoh")
    return ad, bd[:, 0], cd[0]


def _timebase(cfg: PlantConfig) -> np.ndarray:
    n = int(round(cfg.t_end / cfg.dt)) + 1
    return np.arange(n, dtype=float) * cfg.dt


def reference_step(time: np.ndarray, step_time: float) -> np.ndarray:
    return (time >= step_time).astype(float)


def _delay_line(delay_steps: int) -> np.ndarray:
    # An input commanded at index k reaches the no-delay plant at k+delay_steps.
    # Keeping exactly delay_steps previous values makes the timing explicit.
    return np.zeros(max(delay_steps, 1), dtype=float)


def simulate_open_loop(cfg: PlantConfig) -> SimulationResult:
    """Open-loop response to the scaled step u=r/KPR used in Figures 5/6."""
    ad, bd, cd = _discrete_plant(cfg)
    time = _timebase(cfg)
    reference = reference_step(time, cfg.step_time)
    n = len(time)
    delay_steps = int(round(cfg.delay / cfg.dt))
    line = _delay_line(delay_steps)
    state = np.zeros(2, dtype=float)
    y = np.zeros(n, dtype=float)
    u = reference / cfg.kpr
    for k in range(n):
        y[k] = float(cd @ state)
        if k + 1 < n:
            delayed_u = line[0]
            line[:-1] = line[1:]
            line[-1] = u[k]
            state = ad @ state + bd * delayed_u
    return SimulationResult(time, reference, y, u.copy(), u.copy(), np.zeros(n), None, None, "open_loop")


def simulate_closed_loop(
    cfg: PlantConfig,
    ktr: float,
    saturation: Optional[tuple[float, float]] = None,
) -> SimulationResult:
    """Simulate Eqs. (9), (13), (14), and optionally (15)."""
    if ktr <= 0:
        raise ValueError("ktr must be positive")
    if saturation is not None and saturation[0] > saturation[1]:
        raise ValueError("saturation lower bound must not exceed upper bound")

    ad, bd, cd = _discrete_plant(cfg)
    time = _timebase(cfg)
    reference = reference_step(time, cfg.step_time)
    n = len(time)
    delay_steps = int(round(cfg.delay / cfg.dt))
    line = _delay_line(delay_steps)
    state = np.zeros(2, dtype=float)
    y_inf = 0.0
    y = np.zeros(n, dtype=float)
    u = np.zeros(n, dtype=float)
    u_c = np.zeros(n, dtype=float)
    y_inf_trace = np.zeros(n, dtype=float)

    for k in range(n):
        y[k] = float(cd @ state)
        y_inf_trace[k] = y_inf
        # Eq. (14): compare the reference to the estimated final output.
        u_c[k] = y[k] / cfg.kpr + (ktr / cfg.kpr) * (reference[k] - y_inf)
        if saturation is None:
            u[k] = u_c[k]
        else:
            u[k] = np.clip(u_c[k], saturation[0], saturation[1])

        # Eq. (9): use the actual applied input u, not the unlimited u_c.
        if k + 1 < n:
            y_inf += cfg.dt * (cfg.kpr * u[k] - y[k]) / cfg.t_ar
            delayed_u = line[0]
            line[:-1] = line[1:]
            line[-1] = u[k]
            state = ad @ state + bd * delayed_u

    mode = "saturated" if saturation is not None else "unlimited"
    return SimulationResult(time, reference, y, u, u_c, y_inf_trace, ktr, saturation, mode)


def _trapz(values: np.ndarray, time: np.ndarray) -> float:
    # np.trapezoid is unavailable on some older NumPy installations.
    return float(getattr(np, "trapezoid", np.trapz)(values, time))


def metrics(result: SimulationResult, tolerance: float = 0.02) -> dict[str, float | str | None]:
    """Calculate fixed-window comparison metrics after the reference step."""
    t = result.time
    mask = t >= (1.0 if len(t) == 0 else t[np.flatnonzero(result.reference > 0)[0]])
    tm = t[mask]
    ym = result.output[mask]
    rm = result.reference[mask]
    err = rm - ym
    final = float(ym[-1])
    peak = float(np.max(ym))
    ref_final = float(rm[-1]) if len(rm) else 1.0
    overshoot = max(0.0, (peak - ref_final) / max(abs(ref_final), 1e-12) * 100.0)
    in_band = np.abs(err) <= tolerance * max(abs(ref_final), 1.0)
    settling = None
    if len(in_band):
        suffix_ok = np.logical_and.accumulate(in_band[::-1])[::-1]
        candidates = np.flatnonzero(suffix_ok)
        if len(candidates):
            settling = float(tm[candidates[0]])
    return {
        "mode": result.mode,
        "gain": result.gain,
        "saturation": "none" if result.saturation is None else f"[{result.saturation[0]},{result.saturation[1]}]",
        "final_output": final,
        "peak_output": peak,
        "overshoot_percent": float(overshoot),
        "settling_time_2pct": settling,
        "IAE_step_to_end": _trapz(np.abs(err), tm),
        "ISE_step_to_end": _trapz(err**2, tm),
        "max_applied_input": float(np.max(result.input_applied)),
        "max_unlimited_input": float(np.max(result.input_unlimited)),
    }


def _paper_colors() -> dict[str, str]:
    return {
        "open_loop": "#0072BD",
        "KTR=1": "#A64B2A",
        "KTR=3": "#E5A623",
        "KTR=10": "#7E3F98",
        "KTR=30": "#63A63B",
        "reference": "#6EC5E9",
    }


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.7)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_figure(
    open_loop: SimulationResult,
    closed_loop: Sequence[SimulationResult],
    path: Path,
    limited: bool,
) -> None:
    colors = _paper_colors()
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.5), sharex=True, constrained_layout=True)
    ax_y, ax_u = axes
    ax_y.plot(open_loop.time, open_loop.output, color=colors["open_loop"], linestyle="-.", linewidth=1.2, label="open-loop")
    for r in closed_loop:
        key = f"KTR={int(r.gain)}"
        ax_y.plot(r.time, r.output, color=colors[key], linewidth=1.25, label=key)
    ax_y.plot(open_loop.time, open_loop.reference, color=colors["reference"], linestyle="--", linewidth=1.1, label="r")
    ax_y.set_title("Process output", fontsize=12, fontweight="bold")
    ax_y.set_ylabel("output", fontsize=10)
    ax_y.set_ylim(-0.03, 1.10)
    ax_y.legend(ncol=3, fontsize=7.5, frameon=True, facecolor="white", edgecolor="#C8C8C8", loc="lower right")
    _style_axes(ax_y)

    ax_u.plot(open_loop.time, open_loop.input_applied, color=colors["open_loop"], linestyle="-.", linewidth=1.2, label="open-loop")
    for r in closed_loop:
        key = f"KTR={int(r.gain)}"
        ax_u.plot(r.time, r.input_applied, color=colors[key], linewidth=1.25, label=key)
    ax_u.set_title("Process input", fontsize=12, fontweight="bold")
    ax_u.set_ylabel("input", fontsize=10)
    ax_u.set_xlabel("time [s]", fontsize=10)
    if limited:
        ax_u.set_ylim(-0.03, 3.10)
    else:
        # The paper clips the displayed axis to keep the transient readable;
        # the complete unscaled input is retained in the CSV output.
        ax_u.set_ylim(-0.03, 7.0)
    ax_u.legend(ncol=3, fontsize=7.5, frameon=True, facecolor="white", edgecolor="#C8C8C8", loc="upper right")
    _style_axes(ax_u)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_arrays(results: Sequence[SimulationResult], path: Path) -> None:
    payload: dict[str, np.ndarray] = {}
    for r in results:
        label = r.mode if r.gain is None else f"{r.mode}_ktr_{int(r.gain)}"
        payload[f"{label}_time"] = r.time
        payload[f"{label}_reference"] = r.reference
        payload[f"{label}_output"] = r.output
        payload[f"{label}_input"] = r.input_applied
        payload[f"{label}_input_unlimited"] = r.input_unlimited
        payload[f"{label}_y_inf"] = r.y_inf
    np.savez_compressed(path, **payload)


def run(out_dir: Path, cfg: PlantConfig, gains: Sequence[int] = (1, 3, 10, 30)) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    open_loop = simulate_open_loop(cfg)
    unlimited = [simulate_closed_loop(cfg, float(g), None) for g in gains]
    limited = [simulate_closed_loop(cfg, float(g), (0.0, 3.0)) for g in gains]

    plot_figure(open_loop, unlimited, out_dir / "figure5_unlimited.png", limited=False)
    plot_figure(open_loop, limited, out_dir / "figure6_saturated.png", limited=True)
    save_arrays([open_loop, *unlimited, *limited], out_dir / "simulation_arrays.npz")

    all_metrics = [metrics(open_loop), *(metrics(x) for x in unlimited), *(metrics(x) for x in limited)]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)

    # Explicit checks for the two implementation hazards highlighted in the
    # Week 5 feedback: delay timing and the applied (saturated) input path.
    delay_steps = int(round(cfg.delay / cfg.dt))
    nonzero = np.flatnonzero(open_loop.output > 1e-10)
    first_response = float(open_loop.time[nonzero[0]]) if len(nonzero) else None
    checks = {
        "config": asdict(cfg),
        "delay_steps": delay_steps,
        "open_loop_first_nonzero_output_time": first_response,
        "expected_first_nonzero_output_time": cfg.step_time + cfg.delay,
        "open_loop_final_output": float(open_loop.output[-1]),
        "expected_open_loop_final_output": 1.0,
        "limited_input_upper_bound": 3.0,
        "limited_max_inputs": {str(int(r.gain)): float(np.max(r.input_applied)) for r in limited},
        "pass_delay_timing_within_dt": first_response is not None and abs(first_response - (cfg.step_time + cfg.delay)) <= 2 * cfg.dt,
        "pass_limited_input": all(float(np.max(r.input_applied)) <= 3.0 + 1e-12 for r in limited),
        "pass_open_loop_final_within_1pct": abs(float(open_loop.output[-1]) - 1.0) <= 0.01,
    }
    (out_dir / "validation_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--t-end", type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PlantConfig(dt=args.dt, t_end=args.t_end)
    checks = run(args.out, cfg)
    print(json.dumps(checks, indent=2))
    failed = [k for k in ("pass_delay_timing_within_dt", "pass_limited_input", "pass_open_loop_final_within_1pct") if not checks[k]]
    if failed:
        raise SystemExit(f"Validation failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
