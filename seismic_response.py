"""
Seismic Response of a Two-Story Shear Building Model
=====================================================

A two-story building is modelled as a lumped-mass spring-damper system
subjected to earthquake ground acceleration. The spring and damper forces
are nonlinear functions of the inter-story displacement and velocity, so
the governing equations of motion are a pair of coupled, nonlinear 2nd
order ODEs.

This script:
    1. Fits nonlinear spring and damper force-displacement/velocity data
       using a polynomial least-squares regression built from scratch
       (via the normal equations, no numpy.polyfit).
    2. Converts the 2nd order equations of motion into a system of four
       1st order ODEs.
    3. Solves that system with a hand-written 4th order Runge-Kutta (RK4)
       integrator, verified against a known analytical solution.
    4. Performs a timestep-independence study, then compares RK4 against
       a hand-written Forward Euler integrator for both accuracy and
       computational cost.

Author: Charlie Pearson
"""

from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
T = 2.5                        # Earthquake forcing period [s]
A_LIST = [4.4, 16.0]            # Forcing amplitudes to simulate [m/s^2]
T_START, T_END = 0.0, 10.0      # Simulation time window [s]
TSPAN = np.array([T_START, T_END])

M1, M2 = 533.5, 552.5            # First / second floor masses [kg]
K_F, C_F = 456000.0, 68.7        # Foundation stiffness [N/m] & damping [N s/m]
Y0 = np.array([0.0, 0.0, 0.0, 0.0])  # Initial state: [x1, v1, x2, v2]


# ---------------------------------------------------------------------------
# Polynomial least-squares fit (no constant term)
# ---------------------------------------------------------------------------
def polylsq(x, y, order, xlabel="x", ylabel="y", title="", save_as=None):
    """
    Least-squares polynomial fit of y = a1*x + a2*x^2 + ... + a_order*x^order
    (no constant term), solved directly via the normal equations.

    Parameters
    ----------
    x, y : array_like
        Independent / dependent variable data.
    order : int
        Highest power of x included in the fit.
    save_as : str, optional

    Returns
    -------
    a : ndarray
        Fitted coefficients [a1, a2, ..., a_order].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    # Design matrix, columns are x^1, x^2, ..., x^order (no constant column)
    X = np.array([x**j for j in range(1, order + 1)]).T

    # Normal equations: (X^T X) a = X^T y
    G = X.T @ X
    a = np.linalg.solve(G, X.T @ y)

    # Plot data + fit
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = sum(a[i] * x_fit**(i + 1) for i in range(order))

    plt.figure()
    plt.scatter(x, y, color="tab:blue", label="Data")
    plt.plot(x_fit, y_fit, color="tab:red", linewidth=2, label="Least-squares fit")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if save_as:
        plt.savefig(FIG_DIR / save_as, dpi=150, bbox_inches="tight")
    plt.show()

    return a


def fit_spring_and_damper_coefficients():
    """Load force-displacement / force-velocity data and fit the nonlinear
    spring (cubic) and damper (quadratic) models used in the equations of
    motion."""
    spring_data = np.genfromtxt(DATA_DIR / "springforce.csv", delimiter=",")
    dx, Fsp = spring_data[:, 0], spring_data[:, 1]
    k = polylsq(dx, Fsp, order=3, xlabel="Δx (m)", ylabel="Spring force (N)",
                title="Spring force vs. Δx", save_as="spring_force_fit.png")

    damping_data = np.genfromtxt(DATA_DIR / "dampingforce.csv", delimiter=",")
    dv, Fd = damping_data[:, 0], damping_data[:, 1]
    c = polylsq(dv, Fd, order=2, xlabel="Δv (m/s)", ylabel="Damping force (N)",
                title="Damping force vs. Δv", save_as="damping_force_fit.png")

    print("Fitted spring coefficients:", {f"k{i+1}": f"{v:.3e}" for i, v in enumerate(k)})
    print("Fitted damper coefficients:", {f"c{i+1}": f"{v:.3e}" for i, v in enumerate(c)})
    return k, c


# ---------------------------------------------------------------------------
# Equations of motion as a first-order system, RK4 and Euler solvers
# ---------------------------------------------------------------------------
def ground_acceleration(t, period, amplitude):
    """Sinusoidal earthquake ground acceleration, active only for one period."""
    if t <= period:
        return amplitude * np.sin(2 * np.pi / period * t)
    return 0.0


def building_ode(t, y, k, c, T, A):
    """
    Right-hand side of the 4-state ODE system for the two-story building.

    State vector y = [x1, v1, x2, v2], where x1/x2 are floor displacements
    and v1/v2 are floor velocities. The inter-story spring and damper
    forces are nonlinear functions of the relative displacement/velocity.
    """
    dx, dv = y[2] - y[0], y[3] - y[1]
    F_spring = k[0] * dx + k[1] * dx**2 + k[2] * dx**3
    F_damper = c[0] * dv + c[1] * dv**2
    ag = ground_acceleration(t, T, A)

    dy = np.empty(4)
    dy[0] = y[1]
    dy[1] = -ag - (C_F / M1) * y[1] - (K_F / M1) * y[0] + F_spring / M1 + F_damper / M1
    dy[2] = y[3]
    dy[3] = -ag - F_damper / M2 - F_spring / M2
    return dy


def rk4_solve(dydt, tspan, y0, h, *args):
    """Fixed-step 4th order Runge-Kutta integrator for a system of 1st order ODEs."""
    ti, tf = tspan
    n_steps = int(round((tf - ti) / h))
    t = ti + h * np.arange(n_steps + 1)

    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0
    for i in range(n_steps):
        k1 = dydt(t[i], y[i], *args)
        k2 = dydt(t[i] + h / 2, y[i] + h / 2 * k1, *args)
        k3 = dydt(t[i] + h / 2, y[i] + h / 2 * k2, *args)
        k4 = dydt(t[i] + h, y[i] + h * k3, *args)
        y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y


def forward_euler_solve(dydt, tspan, y0, h, *args):
    """Fixed-step Forward Euler integrator, for comparison against RK4."""
    ti, tf = tspan
    n_steps = int(round((tf - ti) / h))
    t = ti + h * np.arange(n_steps + 1)

    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0
    for i in range(n_steps):
        y[i + 1] = y[i] + h * dydt(t[i], y[i], *args)
    return t, y


def verify_solvers_against_known_solution():
    """Sanity check: both integrators should reproduce y = exp(t) for dy/dt = y."""
    exact = lambda t, y: y
    t_rk4, y_rk4 = rk4_solve(exact, TSPAN, [1.0], 0.01)
    t_fe, y_fe = forward_euler_solve(exact, TSPAN, [1.0], 0.01)

    plt.figure()
    plt.plot(t_rk4, np.exp(t_rk4), label="Exact: $y=e^t$", linewidth=2)
    plt.plot(t_rk4, y_rk4, "--", label="RK4")
    plt.plot(t_fe, y_fe, "-.", label="Forward Euler")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("Solver verification against a known analytical solution")
    plt.legend()
    plt.savefig(FIG_DIR / "solver_verification.png", dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_dynamic_response(t, y, A, h, save_as=None):
    """Plot x1, x2, v1, v2 for one simulation run."""
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(9, 6))
    axs[0, 0].plot(t, y[:, 0], label="x1")
    axs[0, 1].plot(t, y[:, 2], label="x2", color="tab:orange")
    axs[1, 0].plot(t, y[:, 1], label="v1")
    axs[1, 1].plot(t, y[:, 3], label="v2", color="tab:orange")

    axs[0, 0].set_ylabel("Displacement (m)")
    axs[1, 0].set_ylabel("Velocity (m/s)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 1].set_xlabel("Time (s)")
    fig.suptitle(f"Dynamic response, A = {A} m/s$^2$, h = {h * 1000:.2f} ms")
    for ax in axs.flat:
        ax.legend()
    plt.tight_layout()
    if save_as:
        plt.savefig(FIG_DIR / save_as, dpi=150, bbox_inches="tight")
    plt.show()


def plot_timestep_independence(results, A, save_as=None):
    """Overlay x1/x2 responses at successively halved timesteps for amplitude A."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for label, t, y in results:
        ax1.plot(t, y[:, 0], label=label)
        ax2.plot(t, y[:, 2], label=label)

    ax1.set(xlabel="Time (s)", ylabel="Displacement (m)", title=f"x1 response, A={A} m/s$^2$")
    ax2.set(xlabel="Time (s)", ylabel="Displacement (m)", title=f"x2 response, A={A} m/s$^2$")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    if save_as:
        plt.savefig(FIG_DIR / save_as, dpi=150, bbox_inches="tight")
    plt.show()


def plot_euler_vs_rk4(t_fe, y_fe, t_rk4, y_rk4, h_fe, h_rk4, save_as=None):
    """Compare Forward Euler and RK4 solutions side by side."""
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(9, 6))
    labels = [("x1", 0), ("x2", 2), ("v1", 1), ("v2", 3)]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (name, col), (r, c) in zip(labels, positions):
        axs[r, c].plot(t_fe, y_fe[:, col], color="tab:blue", label=f"{name} Forward Euler")
        axs[r, c].plot(t_rk4, y_rk4[:, col], color="tab:red", label=f"{name} RK4")
        axs[r, c].legend(fontsize=8)

    axs[0, 0].set_ylabel("Displacement (m)")
    axs[1, 0].set_ylabel("Velocity (m/s)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 1].set_xlabel("Time (s)")
    fig.suptitle(f"Forward Euler (h={h_fe*1000:.3f} ms) vs. RK4 (h={h_rk4*1000:.2f} ms)")
    plt.tight_layout()
    if save_as:
        plt.savefig(FIG_DIR / save_as, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("1. Least-squares fit of nonlinear spring and damper forces")
    print("=" * 70)
    k, c = fit_spring_and_damper_coefficients()

    print("\n" + "=" * 70)
    print("Verifying RK4 / Forward Euler solvers against an analytical solution")
    print("=" * 70)
    verify_solvers_against_known_solution()

    print("\n" + "=" * 70)
    print("2a. Timestep-independence study")
    print("=" * 70)
    h0 = T / 200
    n_refinements = 3
    for A in A_LIST:
        results = []
        for i in range(n_refinements):
            h = h0 / (2 ** i)
            t, y = rk4_solve(building_ode, TSPAN, Y0, h, k, c, T, A)
            results.append((f"h = T/{200 * 2**i}", t, y))
        plot_timestep_independence(results, A, save_as=f"timestep_independence_A{A}.png")

    # From the study above, h = T/400 gives visually converged results
    h_rk4 = T / 400
    print(f"\nSelected timestep-independent step size: h = T/400 = {h_rk4 * 1000:.3f} ms")

    print("\n" + "=" * 70)
    print("2b. Full RK4 dynamic response for each forcing amplitude")
    print("=" * 70)
    for A in A_LIST:
        t, y = rk4_solve(building_ode, TSPAN, Y0, h_rk4, k, c, T, A)
        plot_dynamic_response(t, y, A, h_rk4, save_as=f"dynamic_response_A{A}.png")

    print("\n" + "=" * 70)
    print("2c. Forward Euler vs. RK4: accuracy and computational cost")
    print("=" * 70)
    A_ref = A_LIST[0]
    t_rk4, y_rk4 = rk4_solve(building_ode, TSPAN, Y0, h_rk4, k, c, T, A_ref)

    # Forward Euler needs a much smaller step to remain numerically stable
    h_fe = T / 6400
    t_fe, y_fe = forward_euler_solve(building_ode, TSPAN, Y0, h_fe, k, c, T, A_ref)
    plot_euler_vs_rk4(t_fe, y_fe, t_rk4, y_rk4, h_fe, h_rk4,
                       save_as="euler_vs_rk4_comparison.png")

    t_target = 2 * T
    idx_fe = np.argmin(np.abs(t_fe - t_target))
    idx_rk4 = np.argmin(np.abs(t_rk4 - t_target))
    error_at_2T = np.abs(y_fe[idx_fe] - y_rk4[idx_rk4])
    print(f"Absolute difference between methods at t = 2T (h_FE = {h_fe*1000:.3f} ms):")
    for name, val in zip(["x1", "v1", "x2", "v2"], error_at_2T):
        print(f"  {name}: {val:.5f}")

    # Shrink the Forward Euler step until the error drops below tolerance
    tol = 1e-2
    while (error_at_2T > tol).any():
        h_fe /= 1.05
        t_fe, y_fe = forward_euler_solve(building_ode, TSPAN, Y0, h_fe, k, c, T, A_ref)
        idx_fe = np.argmin(np.abs(t_fe - t_target))
        error_at_2T = np.abs(y_fe[idx_fe] - y_rk4[idx_rk4])

    n_steps_fe = int((T_END - T_START) / h_fe)
    print(f"\nForward Euler step size needed for < {tol} error at all 4 states: "
          f"h = {h_fe * 1000:.3f} ms ({n_steps_fe} total steps)")

    start = time.time()
    forward_euler_solve(building_ode, TSPAN, Y0, h_fe, k, c, T, A_ref)
    t_fe_run = time.time() - start

    start = time.time()
    rk4_solve(building_ode, TSPAN, Y0, h_rk4, k, c, T, A_ref)
    t_rk4_run = time.time() - start

    print(f"Forward Euler runtime at matched accuracy: {t_fe_run:.3f} s")
    print(f"RK4 runtime:                                {t_rk4_run:.3f} s")
    print(
        "\nRK4 is both more accurate and cheaper here: it is 4th-order accurate "
        "per step (vs. 1st-order for Euler), so it needs far fewer RHS "
        "evaluations overall to reach the same accuracy, even though each "
        "RK4 step costs 4 RHS evaluations against Euler's 1."
    )


if __name__ == "__main__":
    main()
