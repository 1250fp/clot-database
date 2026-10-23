# 2-D Layered Platelet–Fibrin Clot Database (`clot-database`)

Ultra-minimalist CUDA generator and database of complete multi-slab 2-D clots ($R = 24\text{ }\mu\text{m}$, domain $48\text{ }\mu\text{m}$, $L_z = 32\text{ }\mu\text{m}$, WLC $N = 1100$, junction degrees $3\text{--}7$ from Michael & Pancaldi 2023 Code Ocean sample 1).

## Seed Scheme (Paper Protocol)
Every slab has its own unique random seed, and every whole clot has a contiguous, non-overlapping block of $M$ seeds:
- **`M = 9`, Clot 1** ($k = 3.5556\text{ }\mu\text{m}$, `8` platelets/slab, `109` junctions/slab): `9` slabs, seeds `4000..4008`
- **`M = 6`, Clot 1** ($k = 5.3333\text{ }\mu\text{m}$, `11` platelets/slab, `163` junctions/slab): `6` slabs, seeds `4009..4014`
- **`M = 3`, Clot 1** ($k = 10.6667\text{ }\mu\text{m}$, `18` platelets/slab, `328` junctions/slab): `3` slabs, seeds `4015..4017`

## Minimal Structure
- **Code (`src/program19_core/`)**: Only 4 Python modules (`config.py`, `state.py`, `geometry.py`, `cuda_mechanics.py`) + 1 CUDA kernel (`mechanics_kernel.cu` + precompiled `sm_120` cubin & PTX). Zero GUI, zero `pip install` calls, only `numpy` required.
- **Database (`database/`)**:
  - `catalog.json`: Master index of all generated clots and non-overlapping seed blocks.
  - `M09_clot01_seeds4000_4008/`: One folder per whole clot containing `clot.json` and **1 self-contained `.npz` file per slab** (`slab_01_seed4000.npz` .. `slab_09_seed4008.npz`), storing the `30 s` trajectory frames ($t = 0, 30, 60, \dots, 300\text{ s}$), initial/final states, backbone, bending, cohesion, and filopodia arrays (`~300–450 KB` per slab).

## Usage
Double-click `run.bat` or run:
```powershell
py generate_clots.py
```
To generate only a specific $M$ on one PC (resumable per slab):
```powershell
py generate_clots.py --M 9
py generate_clots.py --M 6 3
```
