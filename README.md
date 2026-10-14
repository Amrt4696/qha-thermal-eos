# QHA thermal EOS fit

A small Python script for fitting pressure-temperature-volume data from
quasi-harmonic calculations to a finite-strain Debye thermal equation of
state.

The fit is done in two steps:

1. The reference compression curve at $T_0$ is fitted with a
   third-order Birch-Murnaghan equation of state to obtain $V_0$,
   $K_{T0}$, and $K'_{T0}$.
2. Those parameters are held fixed while $\gamma_0$, $q_0$, and
   $\theta_{D0}$ are fitted to the full P-T-V dataset using the
   finite-strain Debye formulation of Stixrude and Lithgow-Bertelloni
   (2005).

The example dataset is MgSiO3 bridgmanite and reproduces the fitting
procedure used in the accompanying study.

## Files

```
qha-thermal-eos/
├── README.md
├── LICENSE
├── requirements.txt
├── fit_thermal_eos.py
├── data/
│   └── bridgmanite_example.csv
└── example_output/
    ├── fit_report.txt
    ├── fit_parameters.csv
    ├── residuals.csv
    ├── correlation_bm3.csv
    └── correlation_thermal.csv
```

## Input

The input is a CSV file with three columns:

```
Pressure_GPa,Temperature_K,Volume_A3_cell
0,0,170.060866
0,100,170.093388
0,200,170.315231
...
```

`Volume_A3_cell` is the equilibrium QHA volume of the cell used in the
input. The number of formula units in that cell is supplied with `--z`.

The example file contains only the QHA points retained for the fit. If
the QHA minimum is outside the sampled volume range, that point should
be removed before fitting.

## Run the example

Create an environment and install the two dependencies:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run:

```
python fit_thermal_eos.py data/bridgmanite_example.csv \
    --z 4 \
    --atoms-per-fu 5 \
    --tref 298 \
    --output output
```

For the supplied bridgmanite example, the fit should return approximately:

```
V0       = 25.69623 cm3/mol
K0       = 218.05 GPa
K0prime  = 4.02097
gamma0   = 1.68707
q0       = 1.94210
thetaD0  = 807.63 K
```

Small differences in the last digits can occur across SciPy versions.

## Model

The Eulerian finite strain is

$$
f=\frac{1}{2}\left[\left(\frac{V_0}{V}\right)^{2/3}-1\right].
$$

The 298 K reference pressure is described by third-order Birch-Murnaghan:

$$
P_{\mathrm{ref}}=
3K_0f(1+2f)^{5/2}
\left[1+\frac{3}{2}(K'_0-4)f\right].
$$

For the thermal part,

$$
a_1=6\gamma_0,
$$

$$
a_2=-12\gamma_0+36\gamma_0^2-18\gamma_0q_0,
$$

$$
\theta(V)=\theta_0
\left(1+a_1f+\frac{1}{2}a_2f^2\right)^{1/2},
$$

and

$$
\gamma(V)=
\frac{(1+2f)(a_1+a_2f)}
{6\left(1+a_1f+\frac{1}{2}a_2f^2\right)}.
$$

The thermal pressure is calculated from the Debye thermal energy relative
to the reference temperature:

$$
P_{\mathrm{th}}(V,T)=
\frac{\gamma(V)}
{V}
\left[U(V,T)-U(V,T_0)\right].
$$

The fitted pressure is

$$
P(V,T)=P_{\mathrm{ref}}(V)+P_{\mathrm{th}}(V,T).
$$

Pressure curves are given equal total weight in the thermal fit so that
a curve with more temperature points does not dominate the optimization.

## Output

The script writes:

- `fit_report.txt` — fitted parameters, errors, fit statistics, and
  leave-one-pressure-out validation
- `fit_parameters.csv` — final parameters and formal 1-sigma uncertainties
- `residuals.csv` — pressure and volume residuals for every input point
- `correlation_bm3.csv` — correlation matrix for $V_0$, $K_0$, and
  $K'_0$
- `correlation_thermal.csv` — correlation matrix for $\gamma_0$,
  $q_0$, and $\theta_{D0}$

The reported uncertainties are formal least-squares uncertainties. They
do not include uncertainty in the underlying electronic-structure or
QHA calculations.

## Using another material

The same script can be used for another QHA dataset. Prepare the same
three-column CSV and specify:

- `--z`: formula units in the input cell
- `--atoms-per-fu`: atoms per formula unit
- `--tref`: reference temperature

For example, MgO in a two-atom primitive cell uses `--z 1 --atoms-per-fu 2`.

## References

- Birch, F. (1947). Finite elastic strain of cubic crystals. Physical
  Review 71, 809-824.
- Stixrude, L., Lithgow-Bertelloni, C. (2005). Thermodynamics of mantle
  minerals I: physical properties. Geophysical Journal International
  162, 610-632.
