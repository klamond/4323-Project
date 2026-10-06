# ECE 4323 Project — Probabilistic Field Coverage with a Flying Camera

Course project for ECE 4323 (Modern Control Systems), University of New Brunswick.
A quadrotor with a downward-facing camera surveys a rectangular ground workspace,
builds a probabilistic confidence map under pose and measurement uncertainty, and
plans its path to reduce map entropy.

The full brief is in [`docs/ECE_4323_Project_Description_Flying_Robot_Camera.pdf`](docs/ECE_4323_Project_Description_Flying_Robot_Camera.pdf).

## Repository layout

```
docs/      Project description and (later) reports / slides
src/       Python source
  workspace.py   Milestone 1: 2D workspace discretization (GridWorkspace)
figures/   Figures kept for reports and presentations
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

Milestone 1 demo (prints the grid summary and saves `milestone1_grid.png`):

```bash
python src/workspace.py
```

The parameters at the bottom of `src/workspace.py` are placeholders; replace them
with the team's values.

## Conventions

- NED frame: x = North, y = East, ground plane at z = 0.
- Altitude `z_t` is passed as a positive height above ground.

## Milestones

| Milestone | Due | Tasks | Status |
|---|---|---|---|
| 1. Setup | Oct 7 | Task 1: workspace discretization, oral demo | In progress |
| 2. Basic operation | Nov 18 | Tasks 2–4: quadrotor simulation, camera model, EKF pose uncertainty, max-fusion, intermediate report | Not started |
| 3. Final demo | Dec 9 | Tasks 5–6: entropy-based path planning, mission completion, demo | Not started |
| 4. Final report | Dec 21 | Final technical report | Not started |
