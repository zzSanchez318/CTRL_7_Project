# CTRL_7 reproduction of Figures 5 and 6

This package reproduces the reference-tracking examples in Figures 5 and 6 of:

> D. Vrancic, P. Bistak, M. Huba, and P. Moura Oliveira, "A New Closed-Loop Control Paradigm Based on Process Moments," *Mathematics*, 2025, 13, 244.

## Target

The simulated process is

`Gp(s) = 1.5 / (1 + 2s + s^2) * exp(-0.4s)`

with `KPR = 1.5`, `tar = 2.4 s`, a reference step at `t = 1 s`, and a 0.4 s pure input delay. The tracking controller follows Eq. (14):

`u_c = y/KPR + (KTR/KPR) * (r - y_inf)`

where Eq. (9) is implemented as

`d(y_inf)/dt = (KPR*u - y)/tar`.

For Figure 6, the applied input is clipped according to Eq. (15), with `umin = 0` and `umax = 3`. The integral uses the applied input after clipping, as required by the paper.

## Run

```bash
python replicate_process_moments.py
```

Optional settings:

```bash
python replicate_process_moments.py --dt 0.001 --t-end 10 --out results
```

The script generates:

- `figure5_unlimited.png` - unlimited control signals;
- `figure6_saturated.png` - control signals limited to `[0, 3]`;
- `metrics.csv` - final output, overshoot, settling time, IAE, ISE, and input peaks;
- `simulation_arrays.npz` - complete time-series arrays;
- `validation_checks.json` - delay, saturation, and open-loop sanity checks.

## Scope and deviations

The paper does not specify the numerical solver, tolerances, or exact plotting settings. This implementation uses an exact zero-order-hold discretization of the second-order plant and an integer-sample delay buffer. The selected target covers reference tracking only; disturbance rejection and model mismatch are not included in Figures 5 and 6.
