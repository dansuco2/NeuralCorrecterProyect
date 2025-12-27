import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn

# --- sympy para generar las derivadas y lambdify (igual que tu implementación) ---
from sympy import symbols, Function, diff, lambdify

# -----------------------
# 1) DEFINICIÓN DE RED (misma arquitectura que usas)
# -----------------------
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        z = self.fc2(self.act(self.fc1(x)))
        return self.ln(x + z)

class CorrectorResNet(nn.Module):
    def __init__(self, in_dim=14, hidden=256, n_blocks=3):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.act = nn.SiLU()
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, 2)
        try:
            nn.init.kaiming_uniform_(self.inp.weight, nonlinearity='linear')
        except:
            pass
    def forward(self, x):
        x = self.act(self.inp(x))
        x = self.blocks(x)
        return self.out(x)

# -----------------------
# 2) PARÁMETROS ALEATORIOS DENTRO DE LOS RANGOS DADOS
# -----------------------
rng = np.random.default_rng()

# Model parameters
a = rng.uniform(0.01, 0.05)      # prey growth rate
b = rng.uniform(0.0005, 0.002)   # predation rate
c = rng.uniform(0.0005, 0.002)   # predator mortality rate
d = rng.uniform(0.01, 0.03)      # conversion efficiency

# Initial conditions
N0 = rng.uniform(50, 150)
P0 = rng.uniform(10, 60)

# h es el paso pequeño de integración que ingresa el usuario
h = delta = 1.2     # <- intervalo de muestreo / "delta" (feature para la red)

t0 = 0.0
tf = 1200.0

# -----------------------
# 3) Construir funciones simbólicas (Taylor6) - igual que en tu ejemplo
# -----------------------
t = symbols('t')
N, P = symbols('N P', cls=Function)
a_sym, b_sym, c_sym, d_sym = symbols('a b c d')

Nt = N(t)
Pt = P(t)

# Definir ecuaciones diferenciales (Lotka-Volterra)
dNdt = Nt * (a_sym - b_sym * Pt)
dPdt = Pt * (c_sym * Nt - d_sym)

# Derivadas sucesivas hasta orden 6 (sustituyendo derivadas por ecuación)
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

# Lambdify (funciones numéricas)
fN  = lambdify((t, Nt, Pt, a_sym, b_sym, c_sym, d_sym), dNdt, 'numpy')
fN2 = lambdify((t, Nt, Pt, a_sym, b_sym, c_sym, d_sym), dNdt_2, 'numpy')
fN3 = lambdify((t, Nt, Pt, a_sym, b_sym, c_sym, d_sym), dNdt_3, 'numpy')
fN4 = lambdify((t, Nt, Pt, a_sym, b_sym, c_sym, d_sym), dNdt_4, 'numpy')
fN5 = lambdify((t, Nt, Pt, a_sym, b_sym, c_sym, d_sym), dNdt_5, 'numpy')
fN6 = lambdify((t, Nt, Pt, a_sym, b_sym, c_sym, d_sym), dNdt_6, 'numpy')

fP  = lambdify((t, Nt, Pt, a_sym, b_sym, c_sym, d_sym), dPdt, 'numpy')
fP2 = lambdify((t, Nt, Pt, a_sym, b_sym, c_sym, d_sym), dPdt_2, 'numpy')
fP3 = lambdify((t, Nt, Pt, a_sym, b_sym, c_sym, d_sym), dPdt_3, 'numpy')
fP4 = lambdify((t, Nt, Pt, a_sym, b_sym, c_sym, d_sym), dPdt_4, 'numpy')
fP5 = lambdify((t, Nt, Pt, a_sym, b_sym, c_sym, d_sym), dPdt_5, 'numpy')
fP6 = lambdify((t, Nt, Pt, a_sym, b_sym, c_sym, d_sym), dPdt_6, 'numpy')

# -----------------------
# 4) Preparar vectores de tiempo de salida (usar delta como intervalo de muestreo)
# -----------------------
if delta <= 0:
    raise ValueError("delta debe ser > 0")

times_out = np.arange(t0, tf + 1e-12, delta)  # tiempos donde guardamos soluciones (lo que verá la gráfica)

# -----------------------
# 5) Integración: Taylor6 con paso interno h, y Euler explícito con mismo paso h
# -----------------------
Nt_taylor = []
Pt_taylor = []
Nt_euler = []
Pt_euler = []

t_current = t0
Nt_val = N0
Pt_val = P0
Nt_eul = N0
Pt_eul = P0

next_index = 0
Nt_taylor.append(Nt_val)
Pt_taylor.append(Pt_val)
Nt_euler.append(Nt_eul)
Pt_euler.append(Pt_eul)

for k in range(1, len(times_out)):
    target_t = times_out[k]
    # Integramos desde t_current hasta target_t con pasos h (el último paso puede ser más pequeño)
    while t_current < target_t - 1e-12:
        remaining = target_t - t_current
        h_step = min(h, remaining)

        # --- Taylor6 (usando derivadas evaluadas en (t_current, Nt_val, Pt_val))
        dN = fN(t_current, Nt_val, Pt_val, a, b, c, d)
        dN2 = fN2(t_current, Nt_val, Pt_val, a, b, c, d)
        dN3 = fN3(t_current, Nt_val, Pt_val, a, b, c, d)
        dN4 = fN4(t_current, Nt_val, Pt_val, a, b, c, d)
        dN5 = fN5(t_current, Nt_val, Pt_val, a, b, c, d)
        dN6 = fN6(t_current, Nt_val, Pt_val, a, b, c, d)

        Nt_val = Nt_val + (
            h_step * dN +
            (h_step**2 / 2) * dN2 +
            (h_step**3 / 6) * dN3 +
            (h_step**4 / 24) * dN4 +
            (h_step**5 / 120) * dN5 +
            (h_step**6 / 720) * dN6
        )

        dP = fP(t_current, Nt_val, Pt_val, a, b, c, d)  # note: for P we used updated Nt_val (consistent con tu código)
        # But to be consistent with your original code, we should evaluate derivatives at the old N,P; follow original:
        # Re-evaluate using previous point (like your example did separately for P):
        # So use previous N,P for P derivatives:
        # To mirror your code exactly, evaluate P derivatives at previous values:
        # (So we recompute using previous value variables)
        # For clarity here we use values before updating Nt_val for P derivatives:
        # => adjust: store prev values and use them for P computation

        # Let's redo to preserve original behaviour:
        # We'll recompute using prev values (previous_Nt, previous_Pt)
        # So adapt loop: store previous values before the Taylor update.

        # (To keep this block concise, next iteration will do correct computation — see below for corrected version.)

        # Euler explicit (with h_step) for the Euler solution (using derivatives at current Euler values)
        dN_e = fN(t_current, Nt_eul, Pt_eul, a, b, c, d)
        dP_e = fP(t_current, Nt_eul, Pt_eul, a, b, c, d)
        Nt_eul = Nt_eul + h_step * dN_e
        Pt_eul = Pt_eul + h_step * dP_e

        t_current += h_step

    # After reaching target_t, append sampled values
    Nt_taylor.append(Nt_val)
    Pt_taylor.append(Pt_val)
    Nt_euler.append(Nt_eul)
    Pt_euler.append(Pt_eul)

# ---------- NOTE ----------
# The above loop preserved the high-level structure but to align exactly with your original implementation
# (where P Taylor used the previous values for derivative evaluation), we provide a clearer, corrected integrator below.
# For robustness and exact match, we'll re-run integration with an implementation that mirrors your earlier code:
# -----------------------

def integrate_taylor6_and_euler(a, b, c, d, N0, P0, t0, tf, h, delta):
    times_out = np.arange(t0, tf + 1e-12, delta)
    Nt_taylor = []
    Pt_taylor = []
    Nt_euler = []
    Pt_euler = []

    t_current = t0
    Nt_val = N0
    Pt_val = P0
    Nt_eul = N0
    Pt_eul = P0

    Nt_taylor.append(Nt_val)
    Pt_taylor.append(Pt_val)
    Nt_euler.append(Nt_eul)
    Pt_euler.append(Pt_eul)

    for k in range(1, len(times_out)):
        target_t = times_out[k]

        previous_Nt = Nt_val
        previous_Pt = Pt_val
        previous_t = t_current

        # integrate in small steps of size h until we reach target_t
        while t_current < target_t - 1e-12:
            remaining = target_t - t_current
            h_step = min(h, remaining)

            N_current = Nt_val
            P_current = Pt_val

            # Taylor6 for N (using current N_current, P_current, evaluated at t_current)
            dN = fN(t_current, N_current, P_current, a, b, c, d)
            dN2 = fN2(t_current, N_current, P_current, a, b, c, d)
            dN3 = fN3(t_current, N_current, P_current, a, b, c, d)
            dN4 = fN4(t_current, N_current, P_current, a, b, c, d)
            dN5 = fN5(t_current, N_current, P_current, a, b, c, d)
            dN6 = fN6(t_current, N_current, P_current, a, b, c, d)

            Nt_val = (
                Nt_val +
                h_step * dN +
                (h_step**2 / 2) * dN2 +
                (h_step**3 / 6) * dN3 +
                (h_step**4 / 24) * dN4 +
                (h_step**5 / 120) * dN5 +
                (h_step**6 / 720) * dN6
            )

            # Taylor6 for P (evaluate derivatives at the previous N and P to match your posted code)
            dP = fP(t_current, N_current, P_current, a, b, c, d)
            dP2 = fP2(t_current, N_current, P_current, a, b, c, d)
            dP3 = fP3(t_current, N_current, P_current, a, b, c, d)
            dP4 = fP4(t_current, N_current, P_current, a, b, c, d)
            dP5 = fP5(t_current, N_current, P_current, a, b, c, d)
            dP6 = fP6(t_current, N_current, P_current, a, b, c, d)

            Pt_val = (
                Pt_val +
                h_step * dP +
                (h_step**2 / 2) * dP2 +
                (h_step**3 / 6) * dP3 +
                (h_step**4 / 24) * dP4 +
                (h_step**5 / 120) * dP5 +
                (h_step**6 / 720) * dP6
            )

            # Euler explicit with same small step h_step (for baseline)
            dN_e = fN(t_current, Nt_eul, Pt_eul, a, b, c, d)
            dP_e = fP(t_current, Nt_eul, Pt_eul, a, b, c, d)
            Nt_eul = Nt_eul + h_step * dN_e
            Pt_eul = Pt_eul + h_step * dP_e

            t_current += h_step

        # compute large-step Euler predictor if you also want residuals (optional)
        interval = t_current - previous_t
        dN_large = fN(previous_t, previous_Nt, previous_Pt, a, b, c, d)
        dP_large = fP(previous_t, previous_Nt, previous_Pt, a, b, c, d)
        Ne_pred_large = previous_Nt + interval * dN_large
        Pe_pred_large = previous_Pt + interval * dP_large
        # residuals (not used here, but available):
        res_N = Nt_val - Ne_pred_large
        res_P = Pt_val - Pe_pred_large

        Nt_taylor.append(Nt_val)
        Pt_taylor.append(Pt_val)
        Nt_euler.append(Nt_eul)
        Pt_euler.append(Pt_eul)

    times_out = np.array(times_out)
    return times_out, np.array(Nt_taylor), np.array(Pt_taylor), np.array(Nt_euler), np.array(Pt_euler)

# Run the integrator
times_out, Nt_taylor, Pt_taylor, Nt_euler, Pt_euler = integrate_taylor6_and_euler(
    a, b, c, d, N0, P0, t0, tf, h, delta
)

# -----------------------
# 6) Evaluar la red neuronal en cada punto de salida
# -----------------------
# Cargamos checkpoint (map to cpu por si acaso)
ckpt_path = "/content/multi_DELTA_resnet_corrector_1.2.pth"
try:
    # FIX: Add weights_only=False to torch.load to bypass the UnpicklingError
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el checkpoint '{ckpt_path}': {e}")

# Instanciar y cargar pesos
model = CorrectorResNet(in_dim=14)
model.load_state_dict(checkpoint['model_state'])
model.eval()

scalers = checkpoint.get('scalers', None)
if scalers is None:
    raise RuntimeError("El checkpoint no contiene 'scalers' con mean_X/std_X/mean_y/std_y")

mean_X = np.array(scalers['mean_X'])
std_X  = np.array(scalers['std_X'])
mean_y = np.array(scalers['mean_y'])
std_y  = np.array(scalers['std_y'])

# Construir matriz de features (N puntos)
# features order: [a,b,c,d,N0,P0,t,Neuler,Peuler,delta, dt_inv,dt2,dt3,dt_log]
dt_inv = 1.0 / delta
dt2 = delta**2
dt3 = delta**3
dt_log = np.log(delta)

X = np.column_stack([
    np.full_like(times_out, a, dtype=float),
    np.full_like(times_out, b, dtype=float),
    np.full_like(times_out, c, dtype=float),
    np.full_like(times_out, d, dtype=float),
    np.full_like(times_out, N0, dtype=float),
    np.full_like(times_out, P0, dtype=float),
    times_out,
    Nt_euler,    # Neuler (baseline) at sample times
    Pt_euler,    # Peuler
    np.full_like(times_out, delta, dtype=float),
    np.full_like(times_out, dt_inv, dtype=float),
    np.full_like(times_out, dt2, dtype=float),
    np.full_like(times_out, dt3, dtype=float),
    np.full_like(times_out, dt_log, dtype=float)
])

# Normalizar con mean_X/std_X guardados
X_norm = (X - mean_X) / (std_X + 1e-12)
X_tensor = torch.tensor(X_norm, dtype=torch.float32)

with torch.no_grad():
    y_pred_norm = model(X_tensor).numpy()

y_pred = y_pred_norm * std_y + mean_y
DeltaN_pred = y_pred[:, 0]
DeltaP_pred = y_pred[:, 1]

# Construir Euler corregido por la red
Nt_nn_corr = Nt_euler + DeltaN_pred
Pt_nn_corr = Pt_euler + DeltaP_pred

# -----------------------
# 7) Calcular MAE y mostrar
# -----------------------
mae_N_nn_vs_taylor = mean_absolute_error(Nt_taylor, Nt_nn_corr)
mae_P_nn_vs_taylor = mean_absolute_error(Pt_taylor, Pt_nn_corr)

mae_N_euler_vs_taylor = mean_absolute_error(Nt_taylor, Nt_euler)
mae_P_euler_vs_taylor = mean_absolute_error(Pt_taylor, Pt_euler)

print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)
print("N0 =", N0)
print("P0 =", P0)
print("MAE (NN vs Taylor) Prey (N):", mae_N_nn_vs_taylor)
print("MAE (Euler vs Taylor) Prey (N):", mae_N_euler_vs_taylor)
print("MAE (NN vs Taylor) Predator (P):", mae_P_nn_vs_taylor)
print("MAE (Euler vs Taylor) Predator (P):", mae_P_euler_vs_taylor)

# -----------------------
# 8) Graficar (2x1 vertical) con los títulos solicitados
# -----------------------
plt.figure(figsize=(9, 7))

# PRESAS
plt.subplot(2, 1, 1)
plt.plot(times_out, Nt_taylor, label='N(t) Taylor', linewidth=2)
plt.plot(times_out, Nt_euler, '--', label='N(t) Euler')
plt.plot(times_out, Nt_nn_corr, label='N(t) NN')
title_n = f"Prey  \u2014  MAE(NN vs Taylor)={mae_N_nn_vs_taylor:.6g}   MAE(Euler vs Taylor)={mae_N_euler_vs_taylor:.6g}"
plt.title(title_n)
plt.xlabel("Time")
plt.ylabel("N(t)")
plt.legend()
plt.grid()

# DEPREDADORES
plt.subplot(2, 1, 2)
plt.plot(times_out, Pt_taylor, label='P(t) Taylor', linewidth=2)
plt.plot(times_out, Pt_euler, '--', label='P(t) Euler')
plt.plot(times_out, Pt_nn_corr, label='P(t) NN')
title_p = f"Predator  \u2014  MAE(NN vs Taylor)={mae_P_nn_vs_taylor:.6g}   MAE(Euler vs Taylor)={mae_P_euler_vs_taylor:.6g}"
plt.title(title_p)
plt.xlabel("Time")
plt.ylabel("P(t)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()