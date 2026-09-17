"""Couette flow between concentric cylinders.

Velocity field, profiles, torque and a full theoretical derivation
that avoids the Navier-Stokes equations, using only symmetry,
mass conservation, torque balance, Newton's law of viscosity
and one geometric fact about shear.
"""
import numpy as np
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

st.set_page_config(page_title="Couette Flow Lab", page_icon="🌀", layout="wide")
st.title("🌀 Couette flow between concentric cylinders")
st.caption(
    "Steady, laminar, azimuthal flow in the annulus — computed analytically, "
    "derived without the Navier–Stokes equations. See the **Theory** tab."
)

# ----------------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Geometry")
    R1 = st.slider("Inner radius R₁ (m)", 0.02, 0.15, 0.05, 0.005)
    R2 = st.slider("Outer radius R₂ (m)", 0.06, 0.25, 0.10, 0.005)
    L = st.slider("Cylinder length L (m)", 0.1, 2.0, 0.5, 0.05)
    st.header("Rotation")
    w1_rpm = st.slider("Inner cylinder ω₁ (rpm)", -60.0, 60.0, 5.0, 0.5)
    w2_rpm = st.slider("Outer cylinder ω₂ (rpm)", -60.0, 60.0, 0.0, 0.5)
    st.header("Fluid")
    mu_vals = list(np.round(np.logspace(-4, 0.3, 30), 6))
    mu_default = mu_vals[int(np.argmin(np.abs(np.array(mu_vals) - 0.05)))]
    mu = st.select_slider("Viscosity μ (Pa·s)", options=mu_vals, value=mu_default)
    rho = st.number_input("Density ρ (kg/m³)", 500.0, 2000.0, 1000.0, 10.0)

if R2 <= R1:
    st.error("Outer radius R₂ must be larger than inner radius R₁.")
    st.stop()

# ----------------------------------------------------------------------------
# Physics core — the analytic Couette solution
# ----------------------------------------------------------------------------
w1 = w1_rpm * 2 * np.pi / 60.0
w2 = w2_rpm * 2 * np.pi / 60.0

B = (w1 - w2) / (1.0 / R1**2 - 1.0 / R2**2)
A = w2 - B / R2**2

r = np.linspace(R1, R2, 400)
omega = A + B / r**2          # angular velocity of fluid, rad/s
vth = r * omega               # azimuthal velocity, m/s
gdot = np.abs(-2.0 * B / r**2)  # shear rate |r dω/dr|, 1/s
tau = mu * gdot               # shear stress magnitude, Pa

T = 4.0 * np.pi * mu * L * B          # motor torque on inner cylinder, N·m
P = T * (w1 - w2)                     # mechanical power dissipated, W
nu = mu / rho
Re = rho * abs(w1) * R1 * (R2 - R1) / mu
Ta = w1**2 * R1 * (R2 - R1) ** 3 / nu**2 if nu > 0 else np.inf
TA_CRIT = 1708.0

# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def vector_field_fig():
    """Top-down view of the annulus with tangential velocity vectors."""
    nr = 7
    na_outer = 24
    rr_in = R1 + 0.07 * (R2 - R1)
    rr_out = R2 - 0.07 * (R2 - R1)
    radii = np.linspace(rr_in, rr_out, nr)

    Xs, Ys, Ls, Sg, Ss = [], [], [], [], []
    for rj in radii:
        na = max(8, int(round(na_outer * rj / R2)))
        th = np.linspace(0, 2 * np.pi, na, endpoint=False)
        vj = rj * (A + B / rj**2)
        Xs.append(rj * np.cos(th))
        Ys.append(rj * np.sin(th))
        Ls.append(th)                       # angles, for tangential direction
        Sg.append(np.full(na, np.sign(vj)))  # sense of rotation
        Ss.append(np.full(na, abs(vj)))       # speed, for colour
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    TH = np.concatenate(Ls)
    SG = np.concatenate(Sg)
    S = np.concatenate(Ss)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.add_patch(Circle((0, 0), R1, color="0.82", zorder=1))
    for rad in radii:
        ax.add_patch(
            Circle((0, 0), rad, fill=False, edgecolor="0.75",
                   lw=0.6, ls=":", zorder=2)
        )
    vmax = S.max()
    if vmax > 0:
        # Explicit arrow lengths in data units (scale_units='xy', scale=1),
        # so visibility does not depend on quiver's auto-scaling.
        Lmax = 0.16 * R2
        Ln = Lmax * S / vmax
        U = -np.sin(TH) * SG * Ln
        V = np.cos(TH) * SG * Ln
        q = ax.quiver(
            X, Y, U, V, S, cmap="plasma",
            scale=1, scale_units="xy", width=0.012,
            headwidth=2.8, headlength=4.0, zorder=4,
        )
        cbar = fig.colorbar(q, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label("speed |v|  (m/s)")
    for rad in (R1, R2):
        ax.add_patch(Circle((0, 0), rad, fill=False, edgecolor="black", lw=2.4, zorder=5))
    ax.set_aspect("equal")
    m = 1.18 * R2
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_xticks([])
    ax.set_yticks([])
    sense = {1: "↺", -1: "↻"}.get(int(np.sign(w1)) if w1 != 0 else 0, "·")
    ax.set_title(
        f"Top view — inner {w1_rpm:+.1f} rpm {sense}   outer {w2_rpm:+.1f} rpm",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def profiles_fig():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    (ax1, ax2), (ax3, ax4) = axes

    ax1.plot(r, vth, lw=2.5, color="#1f77b4")
    ax1.scatter([R1, R2], [R1 * w1, R2 * w2], color="black", zorder=5)
    ax1.set_xlabel("r (m)")
    ax1.set_ylabel("v_θ (m/s)")
    ax1.set_title("Azimuthal velocity profile")
    ax1.grid(alpha=0.3)

    ax2.plot(r, omega * 60 / (2 * np.pi), lw=2.5, color="#d62728")
    ax2.scatter([R1, R2], [w1_rpm, w2_rpm], color="black", zorder=5)
    ax2.set_xlabel("r (m)")
    ax2.set_ylabel("ω (rpm)")
    ax2.set_title("Fluid angular velocity")
    ax2.grid(alpha=0.3)

    ax3.plot(r, tau, lw=2.5, color="#2ca02c")
    ax3.set_xlabel("r (m)")
    ax3.set_ylabel("|τ| (Pa)")
    ax3.set_title("Shear stress magnitude  ∝ 1/r²")
    ax3.grid(alpha=0.3)

    ax4.plot(r, gdot, lw=2.5, color="#9467bd")
    ax4.set_xlabel("r (m)")
    ax4.set_ylabel("|γ̇| = |r dω/dr| (1/s)")
    ax4.set_title("Shear rate magnitude  ∝ 1/r²")
    ax4.grid(alpha=0.3)

    fig.suptitle("Radial profiles across the gap", fontsize=14)
    fig.tight_layout()
    return fig


def shear_rate_fig():
    """Schematic: subtracting the rigid spin leaves pure shear r·dω."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.1))
    for ax in (ax1, ax2):
        ax.set_xlim(-0.4, 4.4)
        ax.set_ylim(-1.15, 2.6)
        ax.set_aspect("equal")
        ax.axis("off")
        # fluid element: y = radial direction, x = azimuthal direction
        rect = plt.Rectangle((0, 0), 3.4, 1.6, fill=False, edgecolor="black", lw=1.6)
        ax.add_patch(rect)
        ax.text(1.7, -0.45, "azimuthal →", ha="center", fontsize=10)
        ax.text(-0.3, 0.8, "radial\n↑", ha="center", fontsize=10)

    ax1.set_title("Lab frame: what you see", fontsize=12)
    ax1.annotate("", xy=(2.0, 0.12), xytext=(0.2, 0.12),
                 arrowprops=dict(arrowstyle="->", lw=2.2, color="#1f77b4"))
    ax1.text(1.1, 0.28, r"$v_\theta(r)$", color="#1f77b4", fontsize=12)
    ax1.annotate("", xy=(2.9, 1.48), xytext=(0.2, 1.48),
                 arrowprops=dict(arrowstyle="->", lw=2.2, color="#d62728"))
    ax1.text(1.55, 1.62, r"$v_\theta(r)+dv_\theta$", color="#d62728", fontsize=12)
    ax1.annotate("", xy=(3.6, 1.0), xytext=(3.6, 0.55),
                 arrowprops=dict(arrowstyle="<->", lw=1.4, color="black"))
    ax1.text(3.75, 0.72, r"$dr$", fontsize=12)
    ax1.text(0.15, 2.05,
             "slip across the gap = $dv_\\theta$\n…but part of it is rigid spin",
             fontsize=10.5)

    ax2.set_title("Co-rotating frame: subtract the spin ω", fontsize=12)
    ax2.plot(0.2, 0.12, "o", color="#1f77b4", ms=9)
    ax2.text(0.32, 0.02, r"$v_\theta-\omega r = 0$", color="#1f77b4", fontsize=12)
    ax2.annotate("", xy=(1.05, 1.48), xytext=(0.2, 1.48),
                 arrowprops=dict(arrowstyle="->", lw=2.4, color="#2ca02c"))
    ax2.text(1.15, 1.55, r"$dv_\theta-\omega\,dr = r\,d\omega$", color="#2ca02c",
             fontsize=12)
    ax2.annotate("", xy=(3.6, 1.0), xytext=(3.6, 0.55),
                 arrowprops=dict(arrowstyle="<->", lw=1.4, color="black"))
    ax2.text(3.75, 0.72, r"$dr$", fontsize=12)
    ax2.annotate("", xy=(0.55, 2.25), xytext=(1.6, 2.25),
                 arrowprops=dict(arrowstyle="->", lw=1.6, color="0.45",
                                 connectionstyle="arc3,rad=0.35"))
    ax2.text(1.75, 2.28, "subtract rigid spin ω", fontsize=10.5, color="0.35")
    ax2.text(0.35, -0.85,
             r"pure shear: slip per unit gap $\;\dot\gamma = r\,d\omega/dr$",
             fontsize=11)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_field, tab_prof, tab_torque, tab_theory = st.tabs(
    ["🌀 Velocity field", "📈 Profiles", "⚙️ Torque & stability", "📐 Theory (no Navier–Stokes)"]
)

with tab_field:
    st.pyplot(vector_field_fig())
    with st.expander("How to read this plot"):
        st.markdown(
            "Top-down view of the annulus. Each arrow is the local fluid velocity: "
            "always tangent to the circle, length and colour encode speed. "
            "The grey disc is the inner cylinder (solid); the outer black ring is the "
            "outer cylinder wall. Note how the arrows shrink outward — the fluid is "
            "dragged by the faster wall and sheared across the gap."
        )

with tab_prof:
    st.pyplot(profiles_fig())
    st.markdown(
        "Black dots mark the no-slip values imposed by the walls. "
        "Stress and shear rate both fall as $1/r^2$ — the same torque crossing "
        "ever-larger cylindrical surfaces (see Theory, step 3)."
    )

with tab_torque:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Motor torque T", f"{T:.4g} N·m")
    c2.metric("Power dissipated", f"{P:.4g} W")
    c3.metric("Max shear stress (inner wall)", f"{tau[0]:.4g} Pa")
    c4.metric("Gap Reynolds number", f"{Re:.4g}")
    st.divider()
    st.subheader("Laminar-assumption monitor")
    st.markdown(
        "The derivation assumes smooth laminar flow. Spin too fast and centrifugal "
        "instability breaks it into **Taylor vortices**. The (narrow-gap) Taylor number "
        f"warns you: critical value ≈ {TA_CRIT:.0f}."
    )
    st.metric("Taylor number Ta = ω₁²R₁d³/ν²", f"{Ta:.4g}")
    if Ta < TA_CRIT:
        st.success("Ta below critical — the laminar Couette profile should hold.")
    else:
        st.warning(
            "Ta exceeds the critical value — in reality Taylor vortices would appear "
            "and this laminar profile would no longer describe the flow."
        )

with tab_theory:
    st.header("Theory: the Couette profile without Navier–Stokes")
    st.markdown(
        "We will earn the velocity profile using only four ingredients: "
        "**symmetry + mass conservation**, **torque balance** (Newton's 2nd law for rotation), "
        "**Newton's law of viscosity**, and **one geometric fact** about shear. "
        "No PDEs. First, the answer, so you know where we're headed:"
    )
    st.latex(r"""\omega(r) = A + \frac{B}{r^2}, \qquad v_\theta(r) = r\,\omega(r)""")
    st.latex(
        r"""B = \frac{\omega_1 - \omega_2}{1/R_1^2 - 1/R_2^2},"""
        r"""\qquad A = \omega_2 - \frac{B}{R_2^2}"""
    )

    st.subheader("Step 1 — Symmetry: the flow has almost no freedom left")
    st.markdown(
        "Steady flow: nothing depends on time. The cylinders are featureless around "
        "the circle, so nothing depends on θ either. Long cylinders: ignore the ends, "
        "so nothing depends on z. The velocity can only be a function of radius: "
        "$(v_r, v_\\theta, v_z)(r)$."
    )
    st.markdown(
        "**Radial velocity must vanish.** Look at the mass flux crossing a cylindrical "
        "surface of radius $r$: $\\dot m = \\rho\\, v_r(r)\\cdot 2\\pi r L$. In steady flow "
        "mass can't accumulate between two such surfaces, so $\\dot m$ is the same at every "
        "$r$. The walls are impermeable ($v_r = 0$ there), hence $\\dot m = 0$, hence "
        "$v_r(r) = 0$ **everywhere**. That was mass conservation — not Navier–Stokes."
    )
    st.markdown(
        "**Axial velocity must vanish.** Nothing drives flow along $z$ (no pressure "
        "gradient, no moving end-walls), so $v_z = 0$."
    )
    st.markdown(
        "Left with a single unknown scalar function — the angular velocity of the fluid:"
    )
    st.latex(r"""\mathbf{v} = v_\theta(r)\,\hat{\boldsymbol\theta}, \qquad \omega(r) = v_\theta(r)/r""")

    st.subheader("Step 2 — Torque balance: torque is a conserved current")
    st.markdown(
        "Isolate a thin cylindrical shell of fluid between $r$ and $r+dr$. The flow is "
        "steady, so the shell's angular momentum isn't changing — the **net torque on it "
        "is zero** (Newton's 2nd law for rotation, $\\tau_{net} = dL/dt = 0$). "
        "Torque flowing in across the inner face must exactly equal torque flowing out "
        "across the outer face."
    )
    st.latex(r"""T(r) = \text{const} = T \quad \text{at every radius } r""")
    st.markdown(
        "Think of torque as a conserved current flowing outward through the fluid. "
        "This step used zero fluid mechanics beyond 'steady' — it is pure rotational statics."
    )

    st.subheader("Step 3 — From torque to stress: geometry does the $1/r^2$")
    st.markdown(
        "Torque = (shear force on the cylindrical surface) × (lever arm):"
    )
    st.latex(
        r"""T = \underbrace{\tau(r)\cdot 2\pi r L}_{\text{shear force}}"""
        r"""\cdot \underbrace{r}_{\text{lever arm}}"""
        r"""\;\;\Longrightarrow\;\; \tau(r) = \frac{T}{2\pi L\,r^2}"""
    )
    st.markdown(
        "Read it physically: the *same* torque $T$ must cross every cylindrical surface, "
        "but surfaces grow ($\\propto r$) and the lever arm grows ($\\propto r$), so the "
        "stress must fall as $1/r^2$ to deliver it. **Two powers of $r$, two geometric "
        "reasons.** The inner wall — smallest area, shortest lever — works the hardest."
    )

    st.subheader("Step 4 — Newton's law of viscosity, and the shear rate, geometrically")
    st.markdown("Newton's law for a Newtonian fluid: shear stress = viscosity × shear rate,")
    st.latex(r"""\tau = \mu\,\dot\gamma""")
    st.markdown(
        "What is the shear rate $\\dot\\gamma$ between two neighbouring cylindrical shells? "
        "Across the gap $dr$ the azimuthal speed changes by $dv_\\theta$ — but **part of that "
        "change is rigid rotation, which distorts nothing**. If the fluid spun like a solid "
        "record at rate $\\omega$, we'd measure $dv_\\theta = \\omega\\,dr$ with zero shear."
    )
    st.markdown(
        "So ride along with a fluid element: subtract the rigid spin $\\omega$ it already has. "
        "Bottom face: $v_\\theta - \\omega r = 0$. Top face: "
        "$(v_\\theta + dv_\\theta) - \\omega(r+dr) = dv_\\theta - \\omega\\,dr "
        "= d(r\\omega) - \\omega\\,dr = r\\,d\\omega$. "
        "What remains is pure sliding — the top face slips over the bottom at speed "
        "$r\\,d\\omega$ across the gap $dr$:"
    )
    st.latex(r"""\dot\gamma = \frac{r\,d\omega}{dr}""")
    st.pyplot(shear_rate_fig())
    st.markdown(
        "Sanity check: solid-body rotation ($d\\omega = 0$) gives $\\dot\\gamma = 0$ and no "
        "stress — exactly right, the fluid is undeformed."
    )

    st.subheader("Step 5 — Assembly: one ODE, zero PDEs")
    st.markdown("Combine steps 3 and 4:")
    st.latex(
        r"""\mu\, r\,\frac{d\omega}{dr} = \frac{T}{2\pi L\,r^2}"""
        r"""\;\;\Longrightarrow\;\; \frac{d\omega}{dr} = \frac{C}{r^3},"""
        r"""\quad C = \frac{T}{2\pi\mu L}"""
    )
    st.markdown("Integrate once:")
    st.latex(r"""\omega(r) = A + \frac{B}{r^2}, \qquad B = -C/2""")
    st.markdown(
        "Two integration constants, two no-slip conditions "
        "$\\omega(R_1) = \\omega_1$, $\\omega(R_2) = \\omega_2$ — giving the $A$, $B$ "
        "boxed at the top. With $v_\\theta = r\\omega$, the profile is fully determined."
    )

    st.subheader("Step 6 — Reading the solution (physical checks)")
    st.markdown(
        """
- **Stress peaks at the inner wall:** $|\\tau(r)| = 2\\mu|B|/r^2$. Same torque, smallest
  surface — the shaft works hardest. (This is why journal bearings fail at the shaft,
  not the housing.)
- **Co-rotation costs nothing:** $\\omega_1 = \\omega_2 \\Rightarrow B = 0
  \\Rightarrow \\dot\\gamma = 0,\\, \\tau = 0,\\, T = 0$. The fluid rides along as a
  rigid body — stirring without shearing.
- **Narrow gap recovers plane Couette flow:** for $d = R_2 - R_1 \\ll R_1$, expanding
  $\\omega(r)$ gives a *linear* profile $v_\\theta(y) \\approx \\omega_1 R_1 +
  (\\omega_2 R_2 - \\omega_1 R_1)\\,y/d$. Curvature of the profile is a finite-gap effect.
- **Viscosity sets the effort, not the shape:** $\\mu$ appears nowhere in $\\omega(r)$
  or $v_\\theta(r)$. Eliminate $T$ between steps 3 and 5 and $\\mu$ cancels — it scales
  both the stress transmitted and the stress the fluid needs, leaving the profile
  untouched. Viscosity controls *how hard you must twist* ($T$, $\\tau$), geometry
  controls *how the fluid moves*.
- **Bonus — pressure, free of charge:** a fluid parcel circling at $v_\\theta$ needs
  centripetal force, supplied radially by pressure. Force balance on the parcel
  (again, no Navier–Stokes):
        """
    )
    st.latex(r"""\frac{dp}{dr} = \rho\,\frac{v_\theta^2}{r}""")
    st.markdown(
        "Pressure rises outward — the outer wall feels the centrifugal push."
    )

    st.subheader("Step 7 — What we assumed, and what Navier–Stokes would add")
    st.markdown(
        """
Laminar, steady, Newtonian ($\\tau \\propto \\dot\\gamma$), no-slip, long cylinders,
incompressible. **Navier–Stokes would give the identical profile** — we used its
rotational content (torque balance) directly, so the PDE has nothing left to add here.

Two remarks for the road:

- Steps 1–3 survive even for **non-Newtonian** fluids: the torque — and hence the
  $1/r^2$ stress distribution — is fixed by statics alone. Only step 4 changes
  ($\\tau(\\dot\\gamma)$ becomes nonlinear), so only the *shape* of the profile changes.
  Rheometers exploit exactly this: measure $T$ vs $\\omega_1$ in a Couette cell and you
  are directly measuring $\\tau(\\dot\\gamma)$.
- Spin too fast and **one** assumption breaks: laminar flow. Centrifugal instability
  grows into Taylor vortices — watch the Taylor-number monitor in the previous tab.
        """
    )

st.caption("Built to be interrogated — try the questions in the Theory tab against the plots.")
