import numpy as np
import pandas as pd
from sympy import symbols, Function, diff, lambdify
from tqdm import tqdm

# ======================================================
# 1. LOAD PARAMETER SETS
# ======================================================
# The file ParameterSet.csv contains multiple combinations of
# Lotka–Volterra parameters and initial conditions:
#   a, b, c, d : system parameters
#   N0, P0     : initial prey and predator populations
parametros_df = pd.read_csv("ParameterSet.csv")
params = parametros_df[['a', 'b', 'c', 'd', 'N0', 'P0']].values

# ======================================================
# 2. TIME DISCRETIZATION SETTINGS
# ======================================================
# n_times : number of output time points per parameter set
# h       : internal integration step size (Δt)
n_times = 1000
h = 1.2  # Large time step used to stress numerical stability

# ======================================================
# 3. SIXTH-ORDER TAYLOR METHOD (SYMBOLIC DERIVATION)
# ======================================================
# Symbolic setup using SymPy to compute high-order derivatives
t = symbols('t')
N, P = symbols('N P', cls=Function)
a, b, c, d = symbols('a b c d')

Nt = N(t)
Pt = P(t)

# Lotka–Volterra system
dNdt = Nt * (a - b * Pt)
dPdt = Pt * (c * Nt - d)

# Successive time derivatives up to order 6
# Each derivative is expressed only in terms of N, P
# by recursively substituting lower-order derivatives
dNdt_2 = diff(dNdt, t).doit().subs({Nt.diff(t): dNdt, Pt.diff(t): dPdt})
dNdt_3 = diff(dNdt_2, t).doit().subs({Nt.diff(t): dNdt, Pt.diff(t): dPdt})
dNdt_4 = diff(dNdt_3, t).doit().subs({Nt.diff(t): dNdt, Pt.diff(t): dPdt})
dNdt_5 = diff(dNdt_4, t).doit().subs({Nt.diff(t): dNdt, Pt.diff(t): dPdt})
dNdt_6 = diff(dNdt_5, t).doit().subs({Nt.diff(t): dNdt, Pt.diff(t): dPdt})

dPdt_2 = diff(dPdt, t).doit().subs({Nt.diff(t): dNdt, Pt.diff(t): dPdt})
dPdt_3 = diff(dPdt_2, t).doit().subs({Nt.diff(t): dNdt, Pt.diff(t): dPdt})
dPdt_4 = diff(dPdt_3, t).doit().subs({Nt.diff(t): dNdt, Pt.diff(t): dPdt})
dPdt_5 = diff(dPdt_4, t).doit().subs({Nt.diff(t): dNdt, Pt.diff(t): dPdt})
dPdt_6 = diff(dPdt_5, t).doit().subs({Nt.diff(t): dNdt, Pt.diff(t): dPdt})

# Convert symbolic expressions into fast NumPy-callable functions
fN  = lambdify((t, Nt, Pt, a, b, c, d), dNdt,  'numpy')
fN2 = lambdify((t, Nt, Pt, a, b, c, d), dNdt_2, 'numpy')
fN3 = lambdify((t, Nt, Pt, a, b, c, d), dNdt_3, 'numpy')
fN4 = lambdify((t, Nt, Pt, a, b, c, d), dNdt_4, 'numpy')
fN5 = lambdify((t, Nt, Pt, a, b, c, d), dNdt_5, 'numpy')
fN6 = lambdify((t, Nt, Pt, a, b, c, d), dNdt_6, 'numpy')

fP  = lambdify((t, Nt, Pt, a, b, c, d), dPdt,  'numpy')
fP2 = lambdify((t, Nt, Pt, a, b, c, d), dPdt_2, 'numpy')
fP3 = lambdify((t, Nt, Pt, a, b, c, d), dPdt_3, 'numpy')
fP4 = lambdify((t, Nt, Pt, a, b, c, d), dPdt_4, 'numpy')
fP5 = lambdify((t, Nt, Pt, a, b, c, d), dPdt_5, 'numpy')
fP6 = lambdify((t, Nt, Pt, a, b, c, d), dPdt_6, 'numpy')

# ======================================================
# 4. DATA GENERATION LOOP
# ======================================================
# For each parameter set:
#   - Integrate using a 6th-order Taylor method (reference solution)
#   - Integrate using explicit Euler (baseline method)
#   - Compute local residuals between Taylor and Euler predictors
data = []

for sample_id, sample in enumerate(tqdm(params, desc="Generating data")):
    a_, b_, c_, d_, N0, P0 = sample

    # Approximate oscillation period of the Lotka–Volterra system
    # Used to automatically define a final integration time
    period = 2 * np.pi / np.sqrt(a_ * d_)
    tf = 20 * period  # Number of oscillation periods to simulate

    times = np.linspace(0, tf, n_times)

    # Initialize Taylor and Euler states
    t_current = 0.0
    Nt_val, Pt_val = N0, P0
    Nt_eul, Pt_eul = N0, P0

    # Store initial condition
    data.append([
        sample_id, a_, b_, c_, d_, N0, P0,
        t_current, Nt_val, Pt_val,
        Nt_eul, Pt_eul,
        0.0, 0.0
    ])

    for i in range(1, n_times):
        next_t = times[i]
        remaining = next_t - t_current

        # Store previous state for residual computation
        previous_Nt, previous_Pt = Nt_val, Pt_val
        previous_t = t_current

        # Internal integration using substeps of size h
        while remaining > 0:
            h_step = min(h, remaining)

            # --- Taylor expansion (order 6) ---
            dN  = fN (t_current, Nt_val, Pt_val, a_, b_, c_, d_)
            dN2 = fN2(t_current, Nt_val, Pt_val, a_, b_, c_, d_)
            dN3 = fN3(t_current, Nt_val, Pt_val, a_, b_, c_, d_)
            dN4 = fN4(t_current, Nt_val, Pt_val, a_, b_, c_, d_)
            dN5 = fN5(t_current, Nt_val, Pt_val, a_, b_, c_, d_)
            dN6 = fN6(t_current, Nt_val, Pt_val, a_, b_, c_, d_)

            Nt_val += (
                h_step * dN +
                (h_step**2 / 2) * dN2 +
                (h_step**3 / 6) * dN3 +
                (h_step**4 / 24) * dN4 +
                (h_step**5 / 120) * dN5 +
                (h_step**6 / 720) * dN6
            )

            dP  = fP (t_current, Nt_val, Pt_val, a_, b_, c_, d_)
            dP2 = fP2(t_current, Nt_val, Pt_val, a_, b_, c_, d_)
            dP3 = fP3(t_current, Nt_val, Pt_val, a_, b_, c_, d_)
            dP4 = fP4(t_current, Nt_val, Pt_val, a_, b_, c_, d_)
            dP5 = fP5(t_current, Nt_val, Pt_val, a_, b_, c_, d_)
            dP6 = fP6(t_current, Nt_val, Pt_val, a_, b_, c_, d_)

            Pt_val += (
                h_step * dP +
                (h_step**2 / 2) * dP2 +
                (h_step**3 / 6) * dP3 +
                (h_step**4 / 24) * dP4 +
                (h_step**5 / 120) * dP5 +
                (h_step**6 / 720) * dP6
            )

            # --- Explicit Euler update (baseline) ---
            Nt_eul += h_step * fN(t_current, Nt_eul, Pt_eul, a_, b_, c_, d_)
            Pt_eul += h_step * fP(t_current, Nt_eul, Pt_eul, a_, b_, c_, d_)

            t_current += h_step
            remaining -= h_step

        # Euler predictor over the full interval (used to define the residual)
        interval = t_current - previous_t
        Ne_pred = previous_Nt + interval * fN(previous_t, previous_Nt, previous_Pt, a_, b_, c_, d_)
        Pe_pred = previous_Pt + interval * fP(previous_t, previous_Nt, previous_Pt, a_, b_, c_, d_)

        # Residuals learned by the neural network
        res_N = Nt_val - Ne_pred
        res_P = Pt_val - Pe_pred

        data.append([
            sample_id, a_, b_, c_, d_, N0, P0,
            t_current, Nt_val, Pt_val,
            Nt_eul, Pt_eul,
            res_N, res_P
        ])

# ======================================================
# 5. SAVE DATASET
# ======================================================
columns = [
    'ID', 'a', 'b', 'c', 'd', 'N0', 'P0',
    't', 'Ntaylor', 'Ptaylor',
    'Neuler', 'Peuler',
    'residual_N', 'residual_P'
]

df = pd.DataFrame(data, columns=columns)
df.to_csv(f"data_h_{h}.csv", index=False)
