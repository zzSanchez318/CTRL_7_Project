# CTRL_7 Full Replication Report

## 1. Target

This implementation reproduces the reference-tracking experiments shown in Figures 5 and 6 of:

D. Vrancic, P. Bistak, M. Huba, and P. Moura Oliveira, *A New Closed-Loop Control Paradigm Based on Process Moments*, Mathematics, 2025, 13, 244.

The selected process is

`Gp(s) = 1.5 / (1 + 2s + s^2) * exp(-0.4s)`.

The paper gives `KPR = 1.5` and `tar = 2.4 s`. The reference is a unit step at `t = 1 s`; the input delay is 0.4 s. The tested tracking gains are `KTR = 1, 3, 10, 30`.

## 2. Controller implementation

The code implements the process-moment estimate from Eq. (9):

`d(y_inf)/dt = (KPR*u - y)/tar`.

The tracking controller from Eq. (14) is:

`u_c = y/KPR + (KTR/KPR)*(r - y_inf)`.

For the limited-input experiment, Eq. (15) is applied:

`u = clip(u_c, 0, 3)`.

The process state is updated with an exact zero-order-hold discretization of the second-order plant. The pure delay is represented by an integer-sample buffer. The process-moment integral always uses the applied input after saturation, rather than the unlimited controller signal.

## 3. Reproduction results

The default run uses `dt = 0.001 s` and ends at `t = 10 s`.

- The open-loop response begins at `1.401 s`; the expected delay-shifted step time is `1.400 s`.
- The open-loop final output is `0.99823`, within 1% of the unit reference.
- The unlimited branch has no overshoot for all four gains in the simulated window. Increasing `KTR` reduces the 2% settling time from about `5.04 s` at `KTR = 3` to `2.95 s` at `KTR = 30`.
- In the saturated branch, the applied input never exceeds 3.0. The `KTR = 10` and `KTR = 30` responses remain stable and settle more slowly than their unlimited counterparts.
- The generated plots reproduce the paper's two main trends: larger tracking gain gives faster reference tracking, and input saturation slows the high-gain responses without causing overshoot or instability.

The complete numerical values are in `results/metrics.csv`; the full time-series arrays are in `results/simulation_arrays.npz`.

## 4. Validation and limitations

`results/validation_checks.json` records automated checks for delay timing, saturation, and the open-loop steady-state gain. The paper does not specify the exact solver, tolerance, or plotting configuration, so the curves should be interpreted as a numerical reproduction of the published method and trends rather than a pixel-identical reconstruction.

The current target is limited to reference tracking in Figures 5 and 6. Disturbance rejection, model mismatch, and the later comparison with other controllers are not part of this reproduction.

## 5. Reproduction command

```bash
python replicate_process_moments.py
```
