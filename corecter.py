import numpy as np
import pandas as pd
from sympy import symbols, Function, diff, lambdify
from tqdm import tqdm

# =========================
# 1. CARGAR PARÁMETROS DESDE ARCHIVO
# =========================
parametros_df = pd.read_csv("ParameterSet.csv")
params = parametros_df[['a', 'b', 'c', 'd', 'N0', 'P0']].values

# =========================
# 2. DEFINIR PASO Y NÚMERO DE PUNTOS DE SALIDA
# =========================
n_times = 1000  # Número fijo de puntos de salida por combinación (incluyendo inicial)
h = 1.2  # Valor de delta (puedes variarlo aquí)

# =========================
# 3. MÉTODO DE TAYLOR DE ORDEN 6 (Simbólico con SymPy)
# =========================
t = symbols('t')
N, P = symbols('N P', cls=Function)
a, b, c, d = symbols('a b c d')

Nt = N(t)
Pt = P(t)

# Definir ecuaciones diferenciales
dNdt = Nt * (a - b * Pt)
dPdt = Pt * (c * Nt - d)

# Derivadas sucesivas hasta orden 6
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

# Convertir a funciones NumPy
fN = lambdify((t, Nt, Pt, a, b, c, d), dNdt, 'numpy')
fN2 = lambdify((t, Nt, Pt, a, b, c, d), dNdt_2, 'numpy')
fN3 = lambdify((t, Nt, Pt, a, b, c, d), dNdt_3, 'numpy')
fN4 = lambdify((t, Nt, Pt, a, b, c, d), dNdt_4, 'numpy')
fN5 = lambdify((t, Nt, Pt, a, b, c, d), dNdt_5, 'numpy')
fN6 = lambdify((t, Nt, Pt, a, b, c, d), dNdt_6, 'numpy')

fP = lambdify((t, Nt, Pt, a, b, c, d), dPdt, 'numpy')
fP2 = lambdify((t, Nt, Pt, a, b, c, d), dPdt_2, 'numpy')
fP3 = lambdify((t, Nt, Pt, a, b, c, d), dPdt_3, 'numpy')
fP4 = lambdify((t, Nt, Pt, a, b, c, d), dPdt_4, 'numpy')
fP5 = lambdify((t, Nt, Pt, a, b, c, d), dPdt_5, 'numpy')
fP6 = lambdify((t, Nt, Pt, a, b, c, d), dPdt_6, 'numpy')

# =========================
# 4. GENERACIÓN DE DATOS
# =========================
data = []

for sample_id, sample in enumerate(tqdm(params, desc="Generando datos con parámetros reales")):
    a_, b_, c_, d_, N0, P0 = sample

    # Calcular período aproximado y ajustar tf automáticamente para ver oscilaciones
    period = 2 * np.pi / np.sqrt(a_ * d_)
    tf = 20 * period  # Ajusta el multiplicador (20) si quieres más/menos períodos visibles

    times = np.linspace(0, tf, n_times)

    t_current = 0.0
    Nt_val = N0
    Pt_val = P0

    Nt_eul = N0
    Pt_eul = P0

    data.append([sample_id, a_, b_, c_, d_, N0, P0, t_current, Nt_val, Pt_val, Nt_eul, Pt_eul, 0, 0])

    for i in range(1, n_times):
        next_t = times[i]
        remaining = next_t - t_current

        previous_Nt = Nt_val
        previous_Pt = Pt_val
        previous_t = t_current

        while remaining > 0:
            h_step = min(h, remaining)
            N_current = Nt_val
            P_current = Pt_val

            # Taylor orden 6 para N
            dN = fN(t_current, N_current, P_current, a_, b_, c_, d_)
            dN2 = fN2(t_current, N_current, P_current, a_, b_, c_, d_)
            dN3 = fN3(t_current, N_current, P_current, a_, b_, c_, d_)
            dN4 = fN4(t_current, N_current, P_current, a_, b_, c_, d_)
            dN5 = fN5(t_current, N_current, P_current, a_, b_, c_, d_)
            dN6 = fN6(t_current, N_current, P_current, a_, b_, c_, d_)
            Nt_val += (
                h_step * dN +
                (h_step**2 / 2) * dN2 +
                (h_step**3 / 6) * dN3 +
                (h_step**4 / 24) * dN4 +
                (h_step**5 / 120) * dN5 +
                (h_step**6 / 720) * dN6
            )

            # Taylor orden 6 para P
            dP = fP(t_current, N_current, P_current, a_, b_, c_, d_)
            dP2 = fP2(t_current, N_current, P_current, a_, b_, c_, d_)
            dP3 = fP3(t_current, N_current, P_current, a_, b_, c_, d_)
            dP4 = fP4(t_current, N_current, P_current, a_, b_, c_, d_)
            dP5 = fP5(t_current, N_current, P_current, a_, b_, c_, d_)
            dP6 = fP6(t_current, N_current, P_current, a_, b_, c_, d_)
            Pt_val += (
                h_step * dP +
                (h_step**2 / 2) * dP2 +
                (h_step**3 / 6) * dP3 +
                (h_step**4 / 24) * dP4 +
                (h_step**5 / 120) * dP5 +
                (h_step**6 / 720) * dP6
            )

            # Euler explícito para la trayectoria global
            dN_e = fN(t_current, Nt_eul, Pt_eul, a_, b_, c_, d_)
            dP_e = fP(t_current, Nt_eul, Pt_eul, a_, b_, c_, d_)
            Nt_eul += h_step * dN_e
            Pt_eul += h_step * dP_e

            t_current += h_step
            remaining -= h_step

        # Calcular el predictor Euler para el intervalo completo (para residual local con paso grande)
        interval = t_current - previous_t
        dN_large = fN(previous_t, previous_Nt, previous_Pt, a_, b_, c_, d_)
        dP_large = fP(previous_t, previous_Nt, previous_Pt, a_, b_, c_, d_)
        Ne_pred_large = previous_Nt + interval * dN_large
        Pe_pred_large = previous_Pt + interval * dP_large

        res_N = Nt_val - Ne_pred_large
        res_P = Pt_val - Pe_pred_large

        data.append([sample_id, a_, b_, c_, d_, N0, P0, t_current, Nt_val, Pt_val, Nt_eul, Pt_eul, res_N, res_P])

# =========================
# 5. GUARDAR RESULTADOS
# =========================
columns = ['ID', 'a', 'b', 'c', 'd', 'N0', 'P0', 't', 'Ntaylor', 'Ptaylor', 'Neuler', 'Peuler', 'residual_N', 'residual_P']
df = pd.DataFrame(data, columns=columns)
df.to_csv(f"data_h_{h}.csv", index=False)