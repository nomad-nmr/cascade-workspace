---
name: cascade-report
description: Build a PDF report comparing CASCADE-2.0 predicted 13C NMR chemical shifts with experimental 13C shifts for a compound, for one structure or several candidate structures ranked by fit - structure colour-coded by prediction confidence, table matched for minimum MAE, and a clear warning when the numbers of predicted and experimental shifts differ. Use this whenever the user wants to check, validate or compare experimental 13C data (from a structure drawing/SVG, a SMILES, a compound name, a paper or supporting-information PDF, or a pasted "13C NMR (126 MHz) δ ..." string) against CASCADE predictions, asks for a "cascade report", or wants to know whether reported 13C data fit a proposed structure or which of several candidate structures/isomers fits best - even if they don't say "report" or "PDF".
---

# CASCADE 13C report

Produces `<Folder>-report.pdf` for one compound: CASCADE-2.0 (Exp22K-FF-GPR, Boltzmann-weighted MMFF
conformer ensemble) predictions vs. experimental 13C shifts — for a single structure, or for several
candidate structures (e.g. isomers proposed for the same data) compared against the same shifts.

Everything heavy is done by an existing script in this workspace — your job is to get the two inputs
right (structure and experimental shifts), run it, and sanity-check the result.

- Script: `cascade_scripts/nmr_report.py` (run from the workspace root, the folder containing `CASCADE-2.0/` and `cascade_scripts/`).
  It re-executes itself inside `cascade_scripts/.venv`, calls `cascade_scripts/predict13c.py`, and
  needs the `CASCADE-2.0` checkout next to `cascade_scripts/`. Do not reinstall or modify these.
- Runtime: ~10 s for a small rigid molecule, up to a minute or two for flexible ones (200 MMFF
  conformers are generated). Use a generous Bash timeout (300000 ms).

## What the report contains (so you can explain it)

With several candidate structures, page 1 is a **summary**: each structure (plain drawing) with its
MAE and max |Δ| (and which carbon), ranked by MAE, the best one outlined in green; a small red note
appears if a structure's count does not match. Then one detail page per structure, in the same
order. With a single structure there is no summary page. Each detail page has:

- Header: SMILES and where it came from, number of conformers, the experimental data line.
- A banner: green "Counts agree" or red **COUNT MISMATCH** if the number of predicted carbon
  environments differs from the number of experimental peaks.
- The structure with each carbon labelled by its number only (shifts are in the table — putting
  them on the drawing made it too crowded), the number coloured by CASCADE confidence (green High /
  amber Moderate / red Low). Carbons are numbered 1, 2, 3 … counting carbons only, in their order
  in the SVG or SMILES; the table's Carbon(s) column uses the same numbers.
- Table: Carbon(s) | Exp δ | Calc δ | Δ calc−exp | σ | Confidence, ordered by **ascending
  experimental shift**. Predicted environments are matched one-to-one to experimental peaks so the
  total |Δ| is minimal (Hungarian assignment). Unmatched lines are marked † in red, listed at the
  bottom and excluded from the MAE.
- MAE and max |Δ|; the row with the largest |Δ| is highlighted (yellow, bold, red outline).
  Notes at the bottom explain matching and confidence.

Symmetry-equivalent carbons (homotopic/enantiotopic, e.g. the two CH3 of a freely rotating
isopropyl on an achiral molecule) are averaged into one environment (`C8,C9`); diastereotopic ones
are kept separate, because that is how many peaks one should expect.

## Workflow

### 1. Make a folder for the compound

One folder per compound (one set of experimental data), e.g. `cascade_scripts/work/<Project>/<Folder>/`,
with a short filesystem-safe name (e.g. `Thymol`, `Compound-5`). The script uses everything in it:

- one or more structures: every `*.svg` and `*.smi` file is a candidate; its file name (without
  extension) is the structure's name in the report, so pick meaningful names (e.g.
  `pyranone.smi`, `furan-isomer.svg`)
- exactly one experimental data file: `*-13C.txt`

The report is `<Folder>-report.pdf`. To compare an alternative structure, just add its `.svg` or
`.smi` to the existing folder and re-run — don't create a new folder or copy the data. A structure
file the user does not want compared must be moved out of the folder.

If the user already created the folder with these files, skip straight to step 4.

### 2. Structure

- **SVG drawing provided** (RDKit-style SVG with `<prefix>:Atom:i` / `<prefix>:Bond:k` ids, any
  drawing scale): leave it as is — the script rebuilds the molecule from it and keeps the drawing's
  layout and atom order. It prints the SMILES it read; check that it is the intended molecule
  (formula, ring sizes) before trusting the numbers.
- **Otherwise write `<Name>.smi`** containing a single SMILES. Sources, in order of preference: a
  SMILES the user gave; a structure from the paper; a systematic name, translated by you.
  Translating a name by hand is error-prone, so verify the SMILES with RDKit before running (use
  `cascade_scripts/.venv/bin/python`): check the molecular formula, the number of carbons, and that
  the substituent positions match the name (and, if available, the reported HRMS formula or the
  1H integration). If the structure is genuinely ambiguous, ask the user rather than guess — a
  confident report on the wrong molecule is worse than no report.
- Carbon numbers in the report count carbons only, in SVG/SMILES order. If the paper uses its own
  numbering (C2, C3', C1''…), tell the user how the two map — it is not shown in the PDF.
- CASCADE-2.0 scope: neutral, single-fragment molecules containing only H, C, N, O, F, P, S, Cl.
  If the compound is outside that (Br, I, B, Si, metals, salts), say so — the script will stop
  with an error.

### 3. Experimental shifts

Write `<Name>-13C.txt` containing the experimental 13C line **verbatim** as published, e.g.

```
13C NMR (126 MHz, CDCl3) δ ppm 55.5 (CH2-OH); 56.5 (OCH3), 110.1 (H2' or H2''), ..., 190.3 (C=O).
```

The parser takes every decimal number after the first `δ`, ignoring anything in parentheses
(assignments, multiplicities). Keeping the original text matters: it is printed in the report so a
reader can check what was compared.

- **Getting it from a PDF** (paper / SI): the `Read` tool cannot render PDFs here (no poppler), so
  extract the text with pypdf in a throwaway environment:
  ```bash
  uv run --no-project --with pypdf python -I -c "import sys; from pypdf import PdfReader; \
  print('\n'.join(f'=== PAGE {i+1} ===\n' + (p.extract_text() or '') for i, p in enumerate(PdfReader(sys.argv[1]).pages)))" "<file.pdf>" > <scratch>/si.txt
  ```
  Then find the compound's characterisation paragraph (search for its number/name and `13C`). The
  13C list often wraps across lines or even pages, and PDF extraction mangles Greek letters and
  primes (δ, α, β, ′ may vanish) — reassemble the full list and re-insert `δ` before the numbers if
  it was lost. Count the values you extracted and compare with the source.
- **Peaks standing for several carbons.** If the bracket says so explicitly ("128.9 (2C)",
  "56.0 (2 × OCH3)"), the script counts the peak for that many carbons automatically.
  If the paper simply lists fewer peaks than there are carbons (e.g. two OCH3 values for four OCH3
  groups — accidental overlap of very similar carbons is common), do **not** decide on your own:
  run the report first so the mismatch is flagged, explain which peaks probably hide two carbons,
  and let the user choose. When the user agrees, add a separate line to `<Name>-13C.txt`, leaving
  the published line untouched:
  ```
  count twice: 56.5, 56.6
  ```
  and re-run. The report then shows those peaks with `*` in the table and states under the
  experimental line that they were counted twice (assumed accidental overlap), so a reader can see
  the data were interpreted.

### 4. Run

```bash
cd <workspace>/cascade_scripts && ./nmr_report.py work/<Project>/<Name>
```

For each structure it prints the SMILES, the counts, MAE and max |Δ|, then the PDF path. It leaves
in the folder `<Folder>-report.pdf` and one `<structure>-cascade2.txt` per structure (raw
predict13c.py output, per-atom and per-environment tables) plus `<structure>-cascade2.csv` (the
per-environment table only).

### 5. Check and summarise

Read `<Name>-cascade2.txt` and the printed numbers rather than assuming success. Then tell the user,
briefly:

- path to the PDF, the MAE, and the number of matched pairs; with several structures, the ranking
  and how big the MAE gap is (a difference of a few tenths of a ppm is not decisive; look at which
  carbons drive it, e.g. the C=O or the ring carbons that differ between the isomers);
- whether counts agree; if not, the likely reason (overlapping/unreported peaks, peaks listed for
  2C, diastereotopic carbons, or a structure problem) and which lines are unmatched;
- any matched pair with |Δ| > ~3 ppm — for CASCADE (test-set MAE ≈ 0.8 ppm) that is worth a look and
  can indicate a misassignment, a typo in the data, or a wrong structure. Because matching is purely
  by value, one or two badly predicted carbons shift the pairing of their neighbours too, producing
  a chain of large Δs (e.g. a C=O predicted 13 ppm too low pushes every line above it up by one).
  Use the paper's own assignments (often given in brackets) to tell a genuine outlier from such a
  knock-on pairing, and say which it is;
- how carbon numbers map to the paper's numbering if you used a `.smi`.

Remember that the matching is a best fit by value, not a verified assignment, and that the σ-based
confidence labels are uncalibrated — present them as a guide.
