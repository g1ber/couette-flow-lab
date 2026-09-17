# 🌀 Couette Flow Lab

Interactive Streamlit app that calculates and visualizes the **velocity profile
between concentric cylinders** (laminar Couette flow) — with a full theoretical
derivation that avoids the Navier–Stokes equations, using only symmetry, mass
conservation, torque balance, Newton's law of viscosity, and one geometric fact
about shear.

## Features

- **Vector field** — top-down view of the annulus with tangential velocity
  arrows, colored by speed
- **Profiles** — vθ(r), ω(r), shear stress and shear rate (both ∝ 1/r²)
- **Torque & stability** — motor torque, power dissipated, max shear stress,
  gap Reynolds number, and a Taylor-number monitor that warns when laminar
  Couette flow would break down into Taylor vortices
- **Theory** — step-by-step derivation with typeset equations, including a
  schematic of the shear-rate geometry and a free pressure profile from radial
  force balance

## Run it

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Controls

| Input | Meaning |
|---|---|
| R₁, R₂ | Inner / outer cylinder radii (m) |
| ω₁, ω₂ | Cylinder rotation rates (rpm, can be negative / counter-rotating) |
| μ, ρ | Fluid viscosity and density |
| L | Cylinder length (for torque / power) |

## The physics in one line

Steady rotation ⇒ torque is the same at every radius; the same torque crossing
ever-larger cylindrical surfaces forces τ ∝ 1/r²; with τ = μ·r·dω/dr this
integrates to **ω(r) = A + B/r²**, with A, B fixed by no-slip at the walls.
