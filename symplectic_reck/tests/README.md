# Tests

From the repository root:

```bash
python -m unittest discover -s symplectic_reck/tests -v
```

Run one group with, for example:

```bash
python -m unittest symplectic_reck.tests.test_conditioning -v
```

The tests check gates and nulling, homodyne noise, conditioning, optical loss,
extra passive gates, and paired statistics. Loop failures show the mode, seed,
or parameter that failed. The conditioning check writes only to a temporary folder.

## Saved reference data

`fixed_seed_reference.npz` records outputs from validated runs, so that any change
in numerical results is caught. The physics checks test separately whether the
results satisfy the expected equations. All matrices use `(x1, p1, x2, p2, ...)` order.

Settings: N=3, target seed 20, squeezing in [0, 0.5], and noise seed 1000.
Feedback methods used three updates; autodiff used two updates per gate and per
single mode. Block and tomography measurements used seed 104 (1000 shots for noisy
measurements). Calibration used error bound 0.001, error seed 10090, measurement
seed 13000, and 10000 shots.

Investigate a mismatch before replacing the reference file. If a change is intended,
check the physics independently and record why the reference changed.
