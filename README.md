# CASCADE workspace

Scripts and a Claude Code skill that wrap around [CASCADE-2.0](https://github.com/asbhd/CASCADE-2.0)
(13C NMR shift prediction, Exp22K-FF-GPR model).

| Path | What it is |
|---|---|
| `cascade_scripts/predict13c.py` | Predict 13C shifts for one SMILES (MMFF conformer ensemble, Boltzmann-weighted) |
| `cascade_scripts/nmr_report.py` | PDF report comparing predicted vs. experimental 13C shifts for one or several candidate structures |
| `cascade_scripts/smoke_test.py` | Checks that the CASCADE install reproduces the reference predictions |
| `.claude/skills/cascade-report/` | Claude Code skill that drives `nmr_report.py` from a drawing, SMILES, name, SI PDF or pasted data |
| `CASCADE-2.0/` | Git submodule pinned to a specific upstream commit. **Not** stored in this repo |

## Requirements

- Linux or WSL2 (the pinned TensorFlow 2.11 has no recent macOS/Windows wheels)
- Git
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (it will fetch Python 3.10 for you)
- ~2 GB free disk space (TensorFlow, model weights)
- Optional: [Claude Code](https://claude.com/claude-code) to use the skill

## Setup

```bash
# 1. Clone with the CASCADE-2.0 submodule
git clone --recurse-submodules <this-repo-url> cascade-workspace
cd cascade-workspace
# (forgot --recurse-submodules?  git submodule update --init)

# 2. Create the Python 3.10 environment next to the scripts
cd cascade_scripts
uv venv --python 3.10 .venv
uv pip install --python .venv -r requirements-v2.txt
uv pip install --python .venv --no-deps kgcnn==2.2.1   # kgcnn must be installed without deps

# 3. Verify
.venv/bin/python smoke_test.py --mode notebook
```

The environment must be at `cascade_scripts/.venv`; the scripts re-launch themselves with that
interpreter, so you can run them directly without activating anything.

## Usage

Run from `cascade_scripts/`.

**Predict shifts for one molecule**

```bash
./predict13c.py 'CC(C)C1CCC(C)CC1O' menthol    # writes menthol.txt (report) and menthol.csv (per-environment table)
```

**Report against experimental data.** Make a folder under `work/` containing one or more
candidate structures (`<Name>.svg` or `<Name>.smi`) and the experimental shifts (`<Folder>-13C.txt`,
e.g. `13C NMR (126 MHz, CDCl3) δ 155.0, 136.5, ...`), then:

```bash
./nmr_report.py work/Thymol           # writes work/Thymol/Thymol-report.pdf
```

`work/Thymol/` is a worked example (inputs plus the resulting report).
See `.claude/skills/cascade-report/SKILL.md` for the exact input conventions.

**With Claude Code.** Start Claude Code in the repository root; the `cascade-report` skill is
loaded automatically. Ask e.g. *"Check the 13C data for compound 5 in this SI PDF against CASCADE"*
or *"Which of these three isomers fits this 13C list best?"*.

`work/` and `results/` are git-ignored: put inputs (including copyrighted SI PDFs) and outputs
there without worrying about committing them.

## Scope and limitations

CASCADE-2.0 handles neutral, single-fragment molecules containing only H, C, N, O, F, P, S, Cl.

## Updating CASCADE-2.0

The submodule is pinned to a commit, so everyone runs the same model. To move to a newer upstream
version:

```bash
git -C CASCADE-2.0 fetch && git -C CASCADE-2.0 checkout <commit-or-tag>
git add CASCADE-2.0 && git commit -m "Bump CASCADE-2.0"
```

Colleagues then run `git submodule update` after pulling.

## Citation

If you use CASCADE-2.0, cite the work listed in its
[README](https://github.com/asbhd/CASCADE-2.0#citation).
