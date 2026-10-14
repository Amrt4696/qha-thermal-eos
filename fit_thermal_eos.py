#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, least_squares


R = 8.31446261815324
A3_PER_CM3MOL = 1.66053906660
GL_X, GL_W = leggauss(64)


def debye_D3(y):
    y = np.asarray(y, float)
    shape = y.shape
    a = y.ravel()
    out = np.empty_like(a)

    small = a < 1e-5
    large = a > 50.0
    mid = ~(small | large)

    ys = a[small]
    out[small] = 1.0 - 3.0 * ys / 8.0 + ys**2 / 20.0 - ys**4 / 1680.0

    yl = a[large]
    out[large] = (np.pi**4 / 5.0) / yl**3

    if np.any(mid):
        ym = a[mid]
        x = 0.5 * ym[:, None] * (GL_X[None, :] + 1.0)
        integrand = x**3 / np.expm1(x)
        integral = 0.5 * ym * np.sum(GL_W[None, :] * integrand, axis=1)
        out[mid] = 3.0 * integral / ym**3

    return out.reshape(shape)


def debye_energy(T, theta, atoms_per_fu):
    T, theta = np.broadcast_arrays(np.asarray(T, float), np.asarray(theta, float))
    U = np.zeros_like(T)
    mask = T > 0.0
    if np.any(mask):
        y = theta[mask] / T[mask]
        U[mask] = 3.0 * atoms_per_fu * R * T[mask] * debye_D3(y)
    return U


def bm3_pressure(cold, V):
    V0, K0, K0p = cold
    V = np.asarray(V, float)
    f = 0.5 * ((V0 / V) ** (2.0 / 3.0) - 1.0)
    return (
        3.0
        * K0
        * f
        * (1.0 + 2.0 * f) ** 2.5
        * (1.0 + 1.5 * (K0p - 4.0) * f)
    )


def eos_pressure(params, V, T, tref, atoms_per_fu):
    V0, K0, K0p, gamma0, q0, theta0 = params
    V, T = np.broadcast_arrays(np.asarray(V, float), np.asarray(T, float))

    f = 0.5 * ((V0 / V) ** (2.0 / 3.0) - 1.0)
    a1 = 6.0 * gamma0
    a2 = -12.0 * gamma0 + 36.0 * gamma0**2 - 18.0 * gamma0 * q0

    r = 1.0 + a1 * f + 0.5 * a2 * f**2
    bad = r <= 0.0
    rs = np.where(bad, 1e-30, r)

    theta = theta0 * np.sqrt(rs)
    gamma = (1.0 + 2.0 * f) * (a1 + a2 * f) / (6.0 * rs)

    pref = (
        3.0
        * K0
        * f
        * (1.0 + 2.0 * f) ** 2.5
        * (1.0 + 1.5 * (K0p - 4.0) * f)
    )

    dU = debye_energy(T, theta, atoms_per_fu) - debye_energy(
        np.full_like(T, tref), theta, atoms_per_fu
    )
    pth = gamma * dU / (V * 1e-6) / 1e9

    return np.where(bad, np.nan, pref + pth)


def read_pvt(path, z):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    required = {"Pressure_GPa", "Temperature_K", "Volume_A3_cell"}

    if data.dtype.names is None or not required.issubset(data.dtype.names):
        raise ValueError(
            "Input CSV must contain: Pressure_GPa, Temperature_K, Volume_A3_cell"
        )

    P = np.atleast_1d(data["Pressure_GPa"])
    T = np.atleast_1d(data["Temperature_K"])
    Vcell = np.atleast_1d(data["Volume_A3_cell"])

    good = np.isfinite(P) & np.isfinite(T) & np.isfinite(Vcell)
    P, T, Vcell = P[good], T[good], Vcell[good]

    groups = []
    for pressure in np.unique(P):
        mask = P == pressure
        order = np.argsort(T[mask])
        t = T[mask][order]
        vcell = Vcell[mask][order]
        vmol = (vcell / z) / A3_PER_CM3MOL

        groups.append(
            {
                "P": float(pressure),
                "T": t,
                "Vcell": vcell,
                "Vmol": vmol,
            }
        )

    return groups


def covariance(fit, residual):
    n, m = len(residual), len(fit.x)
    dof = max(n - m, 1)
    ss = float(np.sum(residual**2))

    jtj = fit.jac.T @ fit.jac
    cov = np.linalg.pinv(jtj) * (ss / dof)
    err = np.sqrt(np.maximum(np.diag(cov), 0.0))

    den = np.outer(err, err)
    corr = np.divide(
        cov,
        den,
        out=np.full_like(cov, np.nan),
        where=den > 0,
    )
    cond = float(np.linalg.cond(jtj))

    return cov, err, corr, cond


def reference_volume(group, tref):
    T = group["T"]
    V = group["Vmol"]
    if len(T) < 2 or not (T[0] <= tref <= T[-1]):
        return np.nan
    return float(np.interp(tref, T, V))


def fit_reference_bm3(groups, tref):
    points = []
    for group in groups:
        v = reference_volume(group, tref)
        if np.isfinite(v):
            points.append((group["P"], v))

    if len(points) < 4:
        raise RuntimeError("Need at least four pressure curves spanning Tref.")

    P = np.array([x[0] for x in points], float)
    V = np.array([x[1] for x in points], float)

    v0_guess = V[np.argmin(np.abs(P))] if np.any(P == 0.0) else np.max(V) * 1.05

    fit = least_squares(
        lambda x: bm3_pressure(x, V) - P,
        [v0_guess, 230.0, 4.0],
        bounds=([0.85 * v0_guess, 50.0, 2.0], [1.20 * v0_guess, 500.0, 8.0]),
        x_scale="jac",
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
        max_nfev=5000,
    )

    residual = bm3_pressure(fit.x, V) - P
    cov, err, corr, cond = covariance(fit, residual)

    return {
        "params": fit.x,
        "stderr": err,
        "corr": corr,
        "cond": cond,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "maxabs": float(np.max(np.abs(residual))),
    }


def thermal_arrays(groups):
    counts = [len(g["T"]) for g in groups]
    nmean = float(np.mean(counts))

    PP, TT, VV, WW = [], [], [], []
    for group in groups:
        n = len(group["T"])
        weight = np.sqrt(nmean / n)

        PP.append(np.full(n, group["P"], float))
        TT.append(group["T"])
        VV.append(group["Vmol"])
        WW.append(np.full(n, weight, float))

    return (
        np.concatenate(PP),
        np.concatenate(TT),
        np.concatenate(VV),
        np.concatenate(WW),
    )


def fit_thermal(groups, cold, tref, atoms_per_fu):
    P, T, V, W = thermal_arrays(groups)
    cold = np.asarray(cold, float)

    starts = (
        [1.6, 2.0, 900.0],
        [1.6, 3.0, 900.0],
        [1.8, 1.5, 1200.0],
        [1.5, 1.0, 1600.0],
        [2.0, 2.5, 700.0],
    )
    lower = np.array([0.1, 0.0, 200.0])
    upper = np.array([4.0, 8.0, 2500.0])

    def full(x):
        return np.r_[cold, x]

    def weighted_residual(x):
        pcalc = eos_pressure(full(x), V, T, tref, atoms_per_fu)
        residual = np.where(np.isfinite(pcalc), pcalc - P, 1e5)
        return residual * W

    best = None
    for x0 in starts:
        fit = least_squares(
            weighted_residual,
            np.array(x0, float),
            bounds=(lower, upper),
            x_scale="jac",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=10000,
        )
        score = float(np.sum(weighted_residual(fit.x) ** 2))
        if best is None or score < best[0]:
            best = (score, fit)

    fit = best[1]
    params = full(fit.x)

    pcalc = eos_pressure(params, V, T, tref, atoms_per_fu)
    residual = pcalc - P
    weighted = residual * W

    cov, err, corr, cond = covariance(fit, weighted)

    return {
        "free": fit.x,
        "full": params,
        "stderr": err,
        "corr": corr,
        "cond": cond,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "maxabs": float(np.max(np.abs(residual))),
    }


def fit_two_stage(groups, tref, atoms_per_fu):
    cold = fit_reference_bm3(groups, tref)
    thermal = fit_thermal(groups, cold["params"], tref, atoms_per_fu)
    return cold, thermal


def solve_volume(params, pressure, temperature, tref, atoms_per_fu, hint=None):
    v0 = float(params[0])
    center = v0 if hint is None or not np.isfinite(hint) else float(hint)

    lo = max(0.35 * v0, 0.55 * center)
    hi = min(1.80 * v0, 1.65 * center)
    if hi <= lo:
        lo, hi = 0.35 * v0, 1.80 * v0

    grid = np.linspace(lo, hi, 700)
    y = eos_pressure(params, grid, np.full_like(grid, temperature), tref, atoms_per_fu) - pressure

    roots = []
    for i in range(len(grid) - 1):
        if not (np.isfinite(y[i]) and np.isfinite(y[i + 1])):
            continue
        if y[i] == 0.0:
            roots.append(grid[i])
        elif y[i] * y[i + 1] < 0.0:
            try:
                root = brentq(
                    lambda v: float(
                        eos_pressure(params, [v], [temperature], tref, atoms_per_fu)[0]
                        - pressure
                    ),
                    grid[i],
                    grid[i + 1],
                    xtol=1e-11,
                    rtol=1e-11,
                    maxiter=200,
                )
                roots.append(root)
            except ValueError:
                pass

    if not roots:
        return np.nan

    return float(min(roots, key=lambda v: abs(v - center)))


def evaluate_groups(params, groups, tref, atoms_per_fu, z):
    rows = []
    dv_all = []

    for group in groups:
        for T, Vmol, Vcell in zip(group["T"], group["Vmol"], group["Vcell"]):
            P = group["P"]
            Pcalc = float(eos_pressure(params, [Vmol], [T], tref, atoms_per_fu)[0])
            dP = Pcalc - P

            Vpred = solve_volume(params, P, T, tref, atoms_per_fu, hint=Vmol)
            if np.isfinite(Vpred):
                Vpred_cell = Vpred * A3_PER_CM3MOL * z
                dV = 100.0 * (Vpred_cell - Vcell) / Vcell
                dv_all.append(dV)
            else:
                Vpred_cell = np.nan
                dV = np.nan

            rows.append((P, T, Vcell, Pcalc, dP, Vpred_cell, dV))

    return rows, np.asarray(dv_all, float)


def lopo(groups, tref, atoms_per_fu, z):
    results = []

    if len(groups) < 5:
        return results

    for held in groups:
        train = [g for g in groups if g["P"] != held["P"]]
        try:
            cold, thermal = fit_two_stage(train, tref, atoms_per_fu)
            rows, dv = evaluate_groups(thermal["full"], [held], tref, atoms_per_fu, z)
            dP = np.array([r[4] for r in rows], float)

            results.append(
                (
                    held["P"],
                    len(rows),
                    float(np.sqrt(np.mean(dP**2))),
                    float(np.mean(np.abs(dP))),
                    float(np.max(np.abs(dP))),
                    float(np.sqrt(np.mean(dv**2))),
                    float(np.mean(np.abs(dv))),
                    float(np.max(np.abs(dv))),
                )
            )
        except Exception:
            results.append((held["P"], 0, *([np.nan] * 6)))

    return results


def write_matrix(path, names, matrix):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter"] + list(names))
        for name, row in zip(names, matrix):
            writer.writerow([name] + [f"{x:.8f}" for x in row])


def main():
    parser = argparse.ArgumentParser(
        description="Fit QHA P-T-V data with a BM3 + finite-strain Debye thermal EOS."
    )
    parser.add_argument("input", help="CSV with Pressure_GPa, Temperature_K, Volume_A3_cell")
    parser.add_argument("--z", type=float, default=4.0, help="Formula units per input cell")
    parser.add_argument(
        "--atoms-per-fu",
        type=float,
        default=5.0,
        help="Atoms per formula unit",
    )
    parser.add_argument("--tref", type=float, default=298.0, help="Reference temperature in K")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    groups = read_pvt(args.input, args.z)
    cold, thermal = fit_two_stage(groups, args.tref, args.atoms_per_fu)
    params = thermal["full"]

    residual_rows, dv = evaluate_groups(
        params, groups, args.tref, args.atoms_per_fu, args.z
    )
    cv = lopo(groups, args.tref, args.atoms_per_fu, args.z)

    parameter_names = ["V0_cm3mol", "K0_GPa", "K0prime", "gamma0", "q0", "thetaD0_K"]
    errors = np.r_[cold["stderr"], thermal["stderr"]]

    with open(outdir / "fit_parameters.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value", "Uncertainty_1sigma"])
        for name, value, error in zip(parameter_names, params, errors):
            writer.writerow([name, f"{value:.10f}", f"{error:.10f}"])

        v0_cell = params[0] * A3_PER_CM3MOL * args.z
        v0_cell_err = errors[0] * A3_PER_CM3MOL * args.z
        writer.writerow(
            ["V0_A3_cell", f"{v0_cell:.10f}", f"{v0_cell_err:.10f}"]
        )

    with open(outdir / "residuals.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Pressure_GPa",
                "Temperature_K",
                "Volume_A3_cell",
                "Pressure_model_GPa",
                "dP_GPa",
                "Volume_model_A3_cell",
                "dV_percent",
            ]
        )
        writer.writerows(residual_rows)

    write_matrix(
        outdir / "correlation_bm3.csv",
        ["V0", "K0", "K0prime"],
        cold["corr"],
    )
    write_matrix(
        outdir / "correlation_thermal.csv",
        ["gamma0", "q0", "thetaD0"],
        thermal["corr"],
    )

    with open(outdir / "fit_report.txt", "w") as f:
        f.write("QHA thermal EOS fit\n")
        f.write("===================\n\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Reference temperature: {args.tref:.1f} K\n")
        f.write(f"Formula units per cell: {args.z:g}\n")
        f.write(f"Atoms per formula unit: {args.atoms_per_fu:g}\n")
        f.write(f"Pressure curves: {len(groups)}\n")
        f.write(f"PVT points: {sum(len(g['T']) for g in groups)}\n\n")

        f.write("Stage 1: reference BM3\n")
        f.write("----------------------\n")
        for name, value, error in zip(
            ["V0 [cm3/mol]", "K0 [GPa]", "K0prime"],
            cold["params"],
            cold["stderr"],
        ):
            f.write(f"{name:15s} = {value:.8f} +/- {error:.8f}\n")
        f.write(
            f"RMSE = {cold['rmse']:.6f} GPa   "
            f"MAE = {cold['mae']:.6f} GPa   "
            f"MAX|dP| = {cold['maxabs']:.6f} GPa\n\n"
        )

        f.write("Stage 2: finite-strain Debye thermal parameters\n")
        f.write("-----------------------------------------------\n")
        for name, value, error in zip(
            ["gamma0", "q0", "thetaD0 [K]"],
            thermal["free"],
            thermal["stderr"],
        ):
            f.write(f"{name:15s} = {value:.8f} +/- {error:.8f}\n")
        f.write(
            f"RMSE = {thermal['rmse']:.6f} GPa   "
            f"MAE = {thermal['mae']:.6f} GPa   "
            f"MAX|dP| = {thermal['maxabs']:.6f} GPa\n\n"
        )

        v0_cell = params[0] * A3_PER_CM3MOL * args.z
        f.write(f"Equivalent V0 = {v0_cell:.8f} A3/cell\n")
        if len(dv):
            f.write(
                f"Training volume RMSE = {np.sqrt(np.mean(dv**2)):.6f} %\n\n"
            )

        if cv:
            f.write("Leave-one-pressure-out validation\n")
            f.write("---------------------------------\n")
            f.write(
                "held_P  N    RMSE_P    MAE_P   MAX_dP   RMSE_dV%  MAE_dV%  MAX_dV%\n"
            )
            for row in cv:
                f.write(
                    f"{row[0]:6.1f} {row[1]:3d} "
                    f"{row[2]:9.6f} {row[3]:8.6f} {row[4]:8.6f} "
                    f"{row[5]:10.6f} {row[6]:9.6f} {row[7]:9.6f}\n"
                )

            good = np.array([x[5] for x in cv if np.isfinite(x[5])])
            if len(good):
                f.write(
                    f"\nMean LOPO volume RMSE = {np.mean(good):.6f} %\n"
                    f"Worst LOPO volume RMSE = {np.max(good):.6f} %\n"
                )

    print("Fit complete")
    print(f"V0 K0 K0' = {params[0]:.8f} {params[1]:.8f} {params[2]:.8f}")
    print(
        f"gamma0 q0 thetaD0 = {params[3]:.8f} {params[4]:.8f} {params[5]:.8f}"
    )
    print(f"Thermal RMSE = {thermal['rmse']:.6f} GPa")
    print(f"Output: {outdir}")


if __name__ == "__main__":
    main()
