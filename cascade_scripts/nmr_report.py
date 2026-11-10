#!/usr/bin/env python3
"""Compare CASCADE-2.0 predicted and experimental 13C shifts in a one-page PDF report.

Usage:
  ./nmr_report.py work/Thymol

The folder must contain:
  <name>.svg      structure drawing (RDKit-style SVG with molSVG0:Atom:i / Bond:k ids), or
  <name>.smi      a SMILES string (used when there is no .svg)
  <name>-13C.txt  experimental shifts, e.g. "13C NMR (126 MHz): δ 20.9, 22.7, ..."

Steps:
  1. Rebuild the molecule from the SVG (atoms, bonds, heteroatom labels, double bonds) -> SMILES.
  2. Run predict13c.py on that SMILES (output saved as <name>-cascade2.txt in the folder).
  3. Match predicted environments to experimental peaks with minimum total |Δ| (Hungarian
     assignment; one-to-one), order by ascending experimental shift, compute the MAE.
  4. Write <name>-report.pdf: structure and table colour-coded by CASCADE confidence,
     with a clear warning if the numbers of predicted and experimental shifts differ.

Carbons in the report are numbered 1, 2, 3 ... in their order in the SVG drawing or SMILES
(carbons only). A '<name>-13C.txt' line 'count twice: 56.5, 56.6' counts those peaks for two carbons.
"""
import glob
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

# Never write .pyc files: CASCADE-2.0 tracks its __pycache__ in git, so imports would dirty the submodule.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
_VENV = os.path.join(HERE, ".venv")
if os.path.realpath(sys.prefix) != os.path.realpath(_VENV):
    _venv_python = os.path.join(_VENV, "bin", "python")
    if not os.path.exists(_venv_python):
        sys.exit(f"Python environment not found: {_venv_python}")
    os.execv(_venv_python, [_venv_python, os.path.abspath(__file__)] + sys.argv[1:])

import numpy as np  # noqa: E402

CONF_COLOURS = {  # strong colour (structure, legend), light tint (table cells)
    "High": ("#2e7d32", "#c8e6c9"),
    "Moderate": ("#ef8f00", "#ffe0a3"),
    "Low": ("#c62828", "#ffcdd2"),
}
WARN = "#c62828"


# --- 1. SVG -> molecule -------------------------------------------------------------------------
def _tag(el):
    return el.tag.split("}")[-1]


def mol_from_svg(path):
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    root = ET.parse(path).getroot()
    atoms, bonds, lines, labels = {}, [], [], []
    for el in root.iter():
        t, eid = _tag(el), el.get("id", "")
        if t == "circle" and ":Atom:" in eid:
            atoms[int(eid.split(":")[-1])] = (float(el.get("cx")), float(el.get("cy")))
        elif t == "line" and ":Bond:" in eid:
            bonds.append(tuple(float(el.get(k)) for k in ("x1", "y1", "x2", "y2")))
        elif t == "line":
            lines.append(tuple(float(el.get(k)) for k in ("x1", "y1", "x2", "y2")))
        elif t == "text" and el.text and el.text.strip():
            fs = float(el.get("font-size", 14))
            # text x,y is the lower-left baseline; centre it roughly
            labels.append((el.text.strip(), float(el.get("x")) + 0.3 * fs, float(el.get("y")) - 0.35 * fs))
    if not atoms:
        raise ValueError(f"no atoms (<prefix>:Atom:i circles) found in {path}")

    def nearest_atom(x, y):
        i = min(atoms, key=lambda a: math.dist(atoms[a], (x, y)))
        return i, math.dist(atoms[i], (x, y))

    # drawing scale: median bond length (SVGs come at very different sizes)
    blen = float(np.median([math.dist(b[:2], b[2:]) for b in bonds])) if bonds else 1.0

    # element symbols: the label closest to an atom; "H" next to a heteroatom is its hydrogen
    elem = {i: "C" for i in atoms}
    for sym, x, y in labels:
        m = re.match(r"[A-Z][a-z]?", sym)
        if not m or m.group(0) == "H":
            continue
        i, d = nearest_atom(x, y)
        if d < 0.35 * blen:
            elem[i] = m.group(0)

    # bond orders: count visible strokes running parallel to and alongside each bond
    def strokes_along(b):
        x1, y1, x2, y2 = b
        L = math.dist((x1, y1), (x2, y2))
        ux, uy = (x2 - x1) / L, (y2 - y1) / L
        n = 0
        for lx1, ly1, lx2, ly2 in lines:
            ll = math.dist((lx1, ly1), (lx2, ly2))
            if ll < 1e-6 or abs((lx2 - lx1) / ll * uy - (ly2 - ly1) / ll * ux) > 0.15:  # not parallel
                continue
            mx, my = (lx1 + lx2) / 2 - x1, (ly1 + ly2) / 2 - y1
            along, perp = mx * ux + my * uy, abs(mx * uy - my * ux)
            if 0 < along < L and perp < 0.25 * L:
                n += 1
        return n

    rw = Chem.RWMol()
    for i in sorted(atoms):
        rw.AddAtom(Chem.Atom(elem[i]))
    for b in bonds:
        a1, _ = nearest_atom(b[0], b[1])
        a2, _ = nearest_atom(b[2], b[3])
        strokes = strokes_along(b)
        order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}[min(max(strokes, 1), 3)]
        rw.AddBond(a1, a2, order)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y) in atoms.items():
        conf.SetAtomPosition(i, Point3D(x / blen * 1.5, -y / blen * 1.5, 0.0))  # keep the drawing's layout
    mol.AddConformer(conf, assignId=True)
    return mol


# --- 2. run predict13c.py ---------------------------------------------------------------------
def run_cascade(smiles, out_txt):
    r = subprocess.run([os.path.join(HERE, "predict13c.py"), smiles, out_txt],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"predict13c.py failed:\n{r.stderr}")
    text = open(out_txt).read()
    envs, in_env = [], False
    for line in text.splitlines():
        if line.startswith("Per carbon environment"):
            in_env = True
            continue
        if in_env:
            parts = line.split()
            if not parts:
                if envs:
                    break
                continue
            if parts[0] == "atoms" or not parts[0][0].isdigit():
                continue
            envs.append({
                "atoms_smiles": [int(a) for a in parts[0].split(",")],
                "n_C": int(parts[1]), "shift": float(parts[2]), "spread": float(parts[3]),
                "sigma": float(parts[4]), "confidence": parts[5],
            })
    n_conf = re.search(r"Conformers used: (\d+)", text)
    return envs, int(n_conf.group(1)) if n_conf else None


# --- 3. experimental shifts and matching ------------------------------------------------------
def read_experimental(path):
    """Return (shifts, multiple, text, overlap_note).

    shifts   : one entry per carbon the experimental data account for (a peak standing for two
               carbons appears twice)
    multiple : set of shift values counted more than once
    A peak counts for n carbons if its bracket says so ("128.9 (2C)", "56.0 (2 x OCH3)") or if the
    file has a line "count twice: 56.5, 56.6" (user's decision, e.g. assumed accidental overlap).
    Other bracketed text (assignments) is ignored."""
    lines = open(path, encoding="utf-8").read().splitlines()
    extra = []
    data_lines = []
    for ln in lines:
        m = re.match(r"\s*count twice\s*:(.*)", ln, re.I)
        if m:
            extra += [float(x) for x in re.findall(r"\d+\.\d+", m.group(1))]
        elif ln.strip():
            data_lines.append(ln.strip())
    text = " ".join(data_lines)
    _, _, body = text.partition("δ")
    body = body or text
    peaks = []  # [value, n]
    for part in re.split(r"(\([^)]*\))", body):
        if part.startswith("("):
            m = re.search(r"\b(\d+)\s*(?:C\b|×|x\s)", part)
            if m and peaks:
                peaks[-1][1] = int(m.group(1))
        else:
            peaks += [[float(x), 1] for x in re.findall(r"\d+\.\d+", part)]
    for v in extra:
        hit = [p for p in peaks if abs(p[0] - v) < 0.051]
        if not hit:
            sys.exit(f"'count twice' value {v} not found in the experimental data")
        hit[0][1] = max(hit[0][1], 2)
    shifts = [v for v, n in peaks for _ in range(n)]
    multiple = {v for v, n in peaks if n > 1}
    note = ("Counted for two or more carbons: " + ", ".join(f"{v:.1f}" for v, n in peaks if n > 1)
            + (" (assumed accidental overlap)" if extra else "")) if multiple else ""
    return shifts, multiple, text, note


def match(calc, exp):
    """One-to-one assignment minimising the total |calc - exp|. Returns list of (i_calc, j_exp)."""
    from scipy.optimize import linear_sum_assignment

    d = np.subtract.outer(np.array(calc), np.array(exp))
    # minimise total |Δ| (the MAE); many pairings can tie on that in 1-D, so break ties by the
    # smallest sum of squares, which picks the order-preserving pairing and avoids crossed pairs
    cost = np.abs(d) + 1e-6 * d ** 2
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


# --- 4. PDF -----------------------------------------------------------------------------------
def structure_png(mol, path, size):
    """Draw the molecule (no highlights) and return the pixel position of every atom and the
    direction pointing away from its neighbours, used to place the colour-coded labels."""
    from rdkit.Chem.Draw import rdMolDraw2D

    d = rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions()
    o.padding = 0.12 if mol.GetNumAtoms() > 20 else 0.2
    o.minFontSize = 18
    o.bondLineWidth = 2
    d.DrawMolecule(mol)
    d.FinishDrawing()
    with open(path, "wb") as f:
        f.write(d.GetDrawingText())
    pos = {a.GetIdx(): np.array([d.GetDrawCoords(a.GetIdx()).x, d.GetDrawCoords(a.GetIdx()).y])
           for a in mol.GetAtoms()}
    # label direction: bisector of the widest free angle between the atom's bonds; on ties
    # (e.g. 120° at a substituted ring carbon) prefer the side facing away from the molecule.
    centre = np.mean(list(pos.values()), axis=0)
    away = {}
    for a in mol.GetAtoms():
        i = a.GetIdx()
        out = pos[i] - centre
        out = out / np.linalg.norm(out) if np.linalg.norm(out) > 1e-6 else np.array([1.0, 0.0])
        angs = sorted(math.atan2(*(pos[n.GetIdx()] - pos[i])[::-1]) for n in a.GetNeighbors())
        if not angs:
            away[i] = out
            continue
        best, best_score = None, -1e9
        for k, a0 in enumerate(angs):
            a1 = angs[(k + 1) % len(angs)] + (2 * math.pi if k + 1 == len(angs) else 0)
            gap = a1 - a0
            mid = a0 + gap / 2
            v = np.array([math.cos(mid), math.sin(mid)])
            score = gap + 0.3 * float(v @ out)
            if score > best_score:
                best, best_score = v, score
        away[i] = best
    b = mol.GetBondWithIdx(0)
    bond_px = float(np.linalg.norm(pos[b.GetBeginAtomIdx()] - pos[b.GetEndAtomIdx()]))
    return pos, away, bond_px


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _draw_structure(fig, rect, mol, tmp_png, labels=None):
    """Draw mol into an axes at rect (figure fractions); optional {atom: (text, confidence)} labels."""
    plt = _plt()
    ax = fig.add_axes(rect)
    w_in, h_in = rect[2] * fig.get_figwidth(), rect[3] * fig.get_figheight()
    pos, away, bond_px = structure_png(mol, tmp_png, size=(int(w_in * 200), int(h_in * 200)))
    ax.imshow(plt.imread(tmp_png))
    os.remove(tmp_png)
    ax.axis("off")
    for i, (text, conf) in (labels or {}).items():
        dx, dy = away[i]
        x, y = pos[i] + away[i] * 0.18 * bond_px
        ax.text(x, y, text, fontsize=9.5 if len(labels) <= 15 else 8.5, weight="bold", color=CONF_COLOURS[conf][0],
                ha="left" if dx > 0.35 else "right" if dx < -0.35 else "center",
                va="top" if dy > 0.35 else "bottom" if dy < -0.35 else "center")
    return ax


def summary_page(title, exp_file, n_exp, results, tmp_png):
    """First page when several candidate structures are compared: structures, MAE, max error."""
    plt = _plt()
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.965, f"{title}: candidate structures vs experimental $^{{13}}$C", fontsize=16, weight="bold")
    fig.text(0.06, 0.945, f"CASCADE-2.0 predictions for {len(results)} structures compared with the same "
                          f"{n_exp} experimental shifts ({exp_file}).\nRanked by MAE; full details for each "
                          f"structure on the following pages.", fontsize=8.5, va="top", linespacing=1.4)
    ranked = sorted(results, key=lambda r: r["mae"])
    n = len(ranked)
    ncol = 1 if n <= 3 else 2
    nrow = math.ceil(n / ncol)
    top, bottom = 0.905, 0.04
    cell_h = (top - bottom) / nrow
    cell_w = 0.92 / ncol
    for k, r in enumerate(ranked):
        row, col = divmod(k, ncol)
        x0 = 0.04 + col * cell_w
        y0 = top - (row + 1) * cell_h
        best = k == 0
        fig.patches.append(plt.Rectangle((x0 + 0.005, y0 + 0.005), cell_w - 0.01, cell_h - 0.01,
                                         transform=fig.transFigure, fill=False, lw=1.5 if best else 0.8,
                                         ec="#2e7d32" if best else "#b0bec5"))
        fig.text(x0 + 0.02, y0 + cell_h - 0.025, f"{k + 1}. {r['name']}" + ("   (lowest MAE)" if best else ""),
                 fontsize=12, weight="bold", color="#2e7d32" if best else "black")
        stats = f"MAE = {r['mae']:.2f} ppm      max |Δ| = {r['maxdev']:.2f} ppm (carbon {r['worst_atoms']})"
        fig.text(x0 + 0.02, y0 + 0.02, stats, fontsize=11, weight="bold")
        if r["mismatch"]:
            fig.text(x0 + cell_w - 0.02, y0 + 0.02, f"count mismatch {r['n_calc']} vs {r['n_exp']}",
                     ha="right", fontsize=9, weight="bold", color=WARN)
        _draw_structure(fig, [x0 + 0.02, y0 + 0.05, cell_w - 0.04, cell_h - 0.09], r["mol"], tmp_png)
    return fig


def detail_page(r, exp_line, overlap_note, tmp_png, page_note=""):
    import textwrap
    from matplotlib.patches import Patch

    plt = _plt()
    rows, mol = r["rows"], r["mol"]
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    y = 0.965
    fig.text(0.06, y, f"{r['name']}: predicted vs experimental $^{{13}}$C shifts", fontsize=16, weight="bold")
    if page_note:
        fig.text(0.94, 0.985, page_note, ha="right", fontsize=8, color="#666666")
    y -= 0.03
    fig.text(0.06, y, f"SMILES (from {r['source']}): {r['smiles']}", fontsize=8.5 if len(r["smiles"]) < 80 else 7,
             family="monospace")
    y -= 0.02
    n_conf = r["n_conf"]
    fig.text(0.06, y, f"Prediction: CASCADE-2.0 Exp22K-FF-GPR, Boltzmann-weighted MMFF ensemble "
                      f"({n_conf} conformer{'s' if n_conf != 1 else ''}); symmetry-equivalent carbons averaged",
             fontsize=8)
    y -= 0.018
    exp_lines = textwrap.wrap(f"Experimental: {' '.join(exp_line.split())}", 150)
    if len(exp_lines) > 4:
        exp_lines = exp_lines[:4]
        exp_lines[-1] += " ..."
    fig.text(0.06, y, "\n".join(exp_lines), fontsize=7, va="top", linespacing=1.35)
    y -= 0.012 * (len(exp_lines) - 1)
    if overlap_note:
        y -= 0.016
        fig.text(0.06, y, overlap_note + "  (marked * in the table)", fontsize=8, va="top", weight="bold",
                 color="#ef8f00")

    y -= 0.04
    banner_y = y
    if r["mismatch"]:
        fig.text(0.5, y, f"COUNT MISMATCH: {r['n_calc']} predicted environments vs {r['n_exp']} experimental peaks.\n"
                         f"Unmatched lines are marked † in the table and excluded from the MAE.",
                 ha="center", va="center", fontsize=10.5, weight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.6", fc=WARN, ec=WARN))
    else:
        fig.text(0.5, y, f"Counts agree: {r['n_calc']} predicted environments = {r['n_exp']} experimental peaks",
                 ha="center", va="center", fontsize=10, color="#2e7d32",
                 bbox=dict(boxstyle="round,pad=0.5", fc="#e8f5e9", ec="#2e7d32"))

    # layout from the bottom up: notes, MAE line, table (height grows with rows), legend, structure
    n_rows = len(rows) + 1
    row_h = 0.0185 if n_rows <= 15 else 0.0135
    tbl_bottom = 0.135
    tbl_top = tbl_bottom + n_rows * row_h
    struct_bottom = tbl_top + 0.045
    struct_top = banner_y - 0.03
    _draw_structure(fig, [0.04, struct_bottom, 0.92, struct_top - struct_bottom], mol, tmp_png, r["labels"])
    fig.legend(handles=[Patch(color=CONF_COLOURS[c][0], label=f"{c} confidence") for c in CONF_COLOURS],
               loc="lower center", bbox_to_anchor=(0.5, tbl_top + 0.017), ncol=3, fontsize=9, frameon=False)
    fig.text(0.5, tbl_top + 0.006, "Labels: carbon number, coloured by prediction confidence", ha="center",
             fontsize=8.5, style="italic")

    # table
    header = ["Carbon(s)", "Exp δ (ppm)", "Calc δ (ppm)", "Δ calc−exp", "σ (ppm)", "Confidence"]
    cells, colours = [], []
    worst = r["worst"]
    for row in rows:
        dagger = " †" if (row["exp"] is None or row["calc"] is None) else ""
        cells.append([
            row["atoms"] or "—",
            (f"{row['exp']:.1f}" + (" *" if row["exp_multi"] else "")) if row["exp"] is not None else "—" + dagger,
            f"{row['calc']:.2f}" if row["calc"] is not None else "—" + dagger,
            f"{row['calc'] - row['exp']:+.2f}" if row["exp"] is not None and row["calc"] is not None else "n/a",
            f"{row['sigma']:.3f}" if row["sigma"] is not None else "—",
            row["confidence"] or "unmatched",
        ])
        tint = CONF_COLOURS[row["confidence"]][1] if row["confidence"] else "#eeeeee"
        row_c = ["white"] * 5 + [tint]
        if dagger:
            row_c = ["#fdecea"] * 5 + [tint]
        if len(colours) == worst:
            row_c = ["#fff176"] * 5 + [tint]  # row with the largest |Δ|
        colours.append(row_c)
    tax = fig.add_axes([0.06, tbl_bottom, 0.88, tbl_top - tbl_bottom])
    tax.axis("off")
    tbl = tax.table(cellText=cells, colLabels=header, cellColours=colours, bbox=[0, 0, 1, 1],
                    cellLoc="center", colColours=["#37474f"] * len(header))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9 if n_rows <= 15 else 7.5)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_edgecolor("#b0bec5")
        if i == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif j == 5:
            cell.get_text().set_weight("bold")
            conf = rows[i - 1]["confidence"]
            cell.get_text().set_color(CONF_COLOURS[conf][0] if conf else "#555555")
        if i > 0 and i - 1 == worst:
            cell.get_text().set_weight("bold")
            cell.set_edgecolor("#c62828")
            cell.set_linewidth(1.5)
        if i > 0 and j != 5 and (rows[i - 1]["exp"] is None or rows[i - 1]["calc"] is None):
            cell.get_text().set_color(WARN)

    fig.text(0.06, 0.105, f"MAE = {r['mae']:.2f} ppm over {r['n_pairs']} matched pairs;  "
                          f"max |Δ| = {r['maxdev']:.2f} ppm (highlighted row)", fontsize=11, weight="bold")
    notes = (
        "Rows are ordered by ascending experimental shift. Predicted environments were assigned to experimental peaks\n"
        "one-to-one so that the total |Δ| (and hence the MAE) is minimal; this is a best-fit assignment, not a verified\n"
        "spectral assignment. σ is the standard deviation of the model's Gaussian-process prediction; confidence uses the\n"
        "CASCADE web-server thresholds on 1.96σ (High < 1.42 ppm ≤ Moderate < 1.49 ppm ≤ Low). Carbons are numbered\n"
        f"1, 2, 3 ... in their order in {r['source']} (carbons only)."
    )
    fig.text(0.06, 0.015, notes, fontsize=7.3, color="#444444", linespacing=1.4)
    return fig


# --- main -------------------------------------------------------------------------------------
def analyse(path, folder, exp, exp_multi):
    """Predict shifts for one candidate structure and match them to the experimental shifts."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    source = os.path.basename(path)
    name = os.path.splitext(source)[0]
    if source.endswith(".svg"):
        mol = mol_from_svg(path)
        smiles = Chem.MolToSmiles(mol)
        order = list(mol.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])  # SMILES pos -> SVG idx
    else:  # first token of the file is the SMILES; numbering = SMILES atom order
        smiles = open(path).read().split()[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            sys.exit(f"invalid SMILES in {source}: {smiles}")
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(mol)
        order = list(range(mol.GetNumAtoms()))
    print(f"{name}: SMILES {smiles}")

    envs, n_conf = run_cascade(smiles, os.path.join(folder, f"{name}-cascade2.txt"))
    for e in envs:
        e["atoms_svg"] = sorted(order[a - 1] for a in e["atoms_smiles"])
    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"]  # source order
    cnum = {idx: k + 1 for k, idx in enumerate(carbons)}  # carbon-only numbering

    pairs = match([e["shift"] for e in envs], exp)
    calc_of_exp = {j: i for i, j in pairs}
    rows = []
    for j in sorted(range(len(exp)), key=lambda k: exp[k]):
        e = envs[calc_of_exp[j]] if j in calc_of_exp else None
        rows.append({"exp": exp[j], "exp_multi": exp[j] in exp_multi, "calc": e["shift"] if e else None,
                     "atoms": ",".join(str(cnum[a]) for a in e["atoms_svg"]) if e else "",
                     "sigma": e["sigma"] if e else None, "confidence": e["confidence"] if e else None})
    used = {i for i, _ in pairs}
    for i in sorted(set(range(len(envs))) - used, key=lambda k: envs[k]["shift"]):
        e = envs[i]
        rows.append({"exp": None, "exp_multi": False, "calc": e["shift"],
                     "atoms": ",".join(str(cnum[a]) for a in e["atoms_svg"]),
                     "sigma": e["sigma"], "confidence": e["confidence"]})
    matched = [k for k, r in enumerate(rows) if r["exp"] is not None and r["calc"] is not None]
    diffs = [abs(rows[k]["calc"] - rows[k]["exp"]) for k in matched]
    worst = max(matched, key=lambda k: abs(rows[k]["calc"] - rows[k]["exp"]))
    res = {
        "name": name, "source": source, "smiles": smiles, "mol": mol, "n_conf": n_conf, "rows": rows,
        "mae": float(np.mean(diffs)), "maxdev": max(diffs), "worst": worst, "worst_atoms": rows[worst]["atoms"],
        "n_pairs": len(diffs), "n_calc": len(envs), "n_exp": len(exp), "mismatch": len(envs) != len(exp),
        "labels": {a: (str(cnum[a]), e["confidence"]) for e in envs for a in e["atoms_svg"]},
    }
    print(f"  {len(envs)} predicted environments vs {len(exp)} experimental shifts; MAE = {res['mae']:.2f} ppm, "
          f"max |Δ| = {res['maxdev']:.2f} ppm" + ("   ** COUNT MISMATCH **" if res["mismatch"] else ""))
    return res


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: nmr_report.py <folder with one or more NAME.svg / NAME.smi and one *-13C.txt>")
    folder = os.path.abspath(sys.argv[1])
    structs = sorted(glob.glob(os.path.join(folder, "*.svg")) + glob.glob(os.path.join(folder, "*.smi")))
    exps = glob.glob(os.path.join(folder, "*-13C.txt"))
    if not structs or len(exps) != 1:
        sys.exit(f"expected at least one .svg/.smi structure and exactly one *-13C.txt in {folder}")
    title = os.path.basename(folder)

    exp, exp_multi, exp_line, overlap_note = read_experimental(exps[0])
    if overlap_note:
        print(overlap_note)
    results = [analyse(p, folder, exp, exp_multi) for p in structs]

    from matplotlib.backends.backend_pdf import PdfPages
    plt = _plt()
    pdf_path = os.path.join(folder, f"{title}-report.pdf")
    tmp_png = os.path.join(folder, f".{title}-structure.png")
    with PdfPages(pdf_path) as pdf:
        if len(results) > 1:
            fig = summary_page(title, os.path.basename(exps[0]), len(exp), results, tmp_png)
            pdf.savefig(fig)
            plt.close(fig)
        ranked = sorted(results, key=lambda r: r["mae"])
        for k, r in enumerate(ranked):
            note = f"structure {k + 1} of {len(ranked)} (ranked by MAE)" if len(ranked) > 1 else ""
            fig = detail_page(r, exp_line, overlap_note, tmp_png, note)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Written to {pdf_path}")


if __name__ == "__main__":
    main()
