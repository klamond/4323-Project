# ECE 4323 Project — Probabilistic Field Coverage with a Flying Camera

Course project for ECE 4323 (Modern Control Systems), University of New Brunswick.
A quadrotor with a downward-facing camera surveys a rectangular ground workspace,
builds a probabilistic confidence map under pose and measurement uncertainty, and
plans its path to reduce map entropy.

The full brief is in [`docs/ECE_4323_Project_Description_Flying_Robot_Camera.pdf`](docs/ECE_4323_Project_Description_Flying_Robot_Camera.pdf).




## Schedule

Each bar has a matching GitHub issue (milestones #1–#4, tasks as sub-issues).
Diamonds mark the deadlines in the brief.

```mermaid
gantt
    title ECE 4323 project schedule (Fall 2026)
    dateFormat YYYY-MM-DD
    axisFormat %b %d
    tickInterval 1week

    section Milestone 1 (20%)
    Task 1 workspace discretization (#6)  :done,   t1,  2026-09-22, 2026-10-05
    Demo & presentation (#7)              :active, p1,  2026-10-01, 2026-10-07
    M1 due                                :milestone, m1, 2026-10-07, 0d

    section Milestone 2 (20%)
    Quadrotor dynamics sim (#8)           :        dyn, 2026-10-08, 2026-10-21
    Nested-loop PD control (#9)           :        ctl, 2026-10-15, 2026-10-28
    Task 2 camera model (#10)             :        t2,  2026-10-22, 2026-11-01
    Task 3 EKF & pose uncertainty (#11)   :        t3,  2026-10-27, 2026-11-11
    Task 4 max-fusion & map (#12)         :        t4,  2026-11-04, 2026-11-14
    Intermediate report (#5)              :        r2,  2026-11-09, 2026-11-18
    M2 due                                :milestone, m2, 2026-11-18, 0d

    section Milestone 3 (20%)
    Task 5 entropy path planner (#13)     :        t5,  2026-11-19, 2026-12-02
    Task 6 completion criteria (#14)      :        t6,  2026-11-26, 2026-12-04
    Moving target demo (#15)              :        mt,  2026-11-30, 2026-12-06
    Final demo & presentation (#16)       :        p3,  2026-12-03, 2026-12-09
    M3 due                                :milestone, m3, 2026-12-09, 0d

    section Milestone 4 (40%)
    Final report (#4)                     :        r4,  2026-11-25, 2026-12-21
    M4 due                                :milestone, m4, 2026-12-21, 0d
```

| Milestone | Due | Deliverables | Weight |
|---|---|---|---|
| 1. Setup | Oct 7 | Task 1: workspace discretization, oral demo | 20% |
| 2. Basic operation | Nov 18 | Tasks 2–4: quadrotor simulation, camera model, EKF pose uncertainty, max-fusion, intermediate report | 20% |
| 3. Final demo | Dec 9 | Tasks 5–6: entropy-based path planning, mission completion, moving-target demo, presentation | 20% |
| 4. Final report | Dec 21 | Final technical report | 40% |

Start dates are suggestions; deadlines come from the brief.
