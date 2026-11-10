#!/usr/bin/env python3
"""Predict 13C NMR shifts for one molecule with CASCADE-2.0 (MMFF conformer ensemble).

Usage:
  ./predict13c.py 'SMILES' [output.txt]

The results are printed and also written to a text file: output.txt if given,
otherwise cascade2_<date-time>.txt; both relative to the folder you run it from.

Two tables are reported:
  * per carbon environment: symmetry-equivalent carbons (homotopic or enantiotopic,
    i.e. giving one peak in an achiral solvent) averaged into one line;
    diastereotopic carbons (e.g. the isopropyl CH3 groups of menthol) are kept separate;
  * per carbon atom: the raw CASCADE prediction for every carbon.
"""
import datetime
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Re-run with the .venv next to this script (so the folder can be renamed or moved).
_VENV = os.path.join(HERE, ".venv")
if os.path.realpath(sys.prefix) != os.path.realpath(_VENV):
    _venv_python = os.path.join(_VENV, "bin", "python")
    if not os.path.exists(_venv_python):
        sys.exit(f"Python environment not found: {_venv_python}")
    os.execv(_venv_python, [_venv_python, os.path.abspath(__file__)] + sys.argv[1:])


# --- symmetry-equivalent carbons ------------------------------------------------------------
def _labelled_smiles(mol3d, idx, mirror=False):
    """Canonical isomeric SMILES with atom idx labelled 13C, stereo taken from the 3D geometry.
    With mirror=True all tetrahedral centres are inverted (the mirror image)."""
    from rdkit import Chem

    m = Chem.Mol(mol3d)
    m.GetAtomWithIdx(int(idx)).SetIsotope(13)
    Chem.AssignStereochemistryFrom3D(m)
    if mirror:
        for a in m.GetAtoms():
            if a.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif a.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return Chem.MolToSmiles(Chem.RemoveHs(m))


def _interchangeable(mol3d, i, j):
    """True if carbons i and j are homotopic or enantiotopic (one NMR signal in an achiral
    solvent); False if diastereotopic. Labelling i or j must give the same molecule or its
    mirror image."""
    smi_j = _labelled_smiles(mol3d, j)
    return smi_j in (_labelled_smiles(mol3d, i), _labelled_smiles(mol3d, i, mirror=True))


def equivalence_groups(mol3d, carbon_idx):
    """Group carbon indices into NMR-equivalent sets, in order of first appearance."""
    from rdkit import Chem

    # Graph symmetry only; stereo (meso mirror halves, diastereotopic groups) is decided by
    # the 3D labelling test in _interchangeable.
    ranks = list(Chem.CanonicalRankAtoms(mol3d, breakTies=False, includeChirality=False))
    groups = []  # list of lists of atom indices
    for idx in carbon_idx:
        for g in groups:
            if ranks[g[0]] == ranks[idx] and _interchangeable(mol3d, g[0], idx):
                g.append(int(idx))
                break
        else:
            groups.append([int(idx)])
    return groups


def merge_table(table, groups, confidence_fn):
    import pandas as pd

    rows = []
    for g in groups:
        sub = table[table.atom_index.isin(g)]
        sigma = sub.sigma_ppm.mean()
        rows.append({
            "atoms": ",".join(str(a + 1) for a in g),
            "n_C": len(g),
            "shift_ppm": round(sub.shift_ppm.mean(), 2),
            "spread_ppm": round(sub.shift_ppm.max() - sub.shift_ppm.min(), 2),
            "sigma_ppm": round(sigma, 3),
            "confidence": confidence_fn(sigma),
        })
    return pd.DataFrame(rows)


# --- main -------------------------------------------------------------------------------------
def main():
    if len(sys.argv) not in (2, 3):
        print("usage: predict13c.py 'SMILES' [output.txt]", file=sys.stderr)
        sys.exit(2)
    smiles = sys.argv[1]
    if len(sys.argv) == 3:
        out_path = os.path.abspath(sys.argv[2])  # resolve before smoke_test changes the cwd
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.abspath(f"cascade2_{stamp}.txt")  # current folder, like a given name

    # Hide TensorFlow's warning noise (C++ and Python) while the model loads and runs.
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        import smoke_test as st
        from rdkit import Chem

        mol, err = st.validate(smiles)
        if mol is None:
            raise ValueError(err)
        idx = st.carbon_indices(Chem.AddHs(mol))
        confs = st.embed_ensemble(mol)
        if not confs:
            raise RuntimeError("no conformers could be generated")
        preds = st.Cascade2().predict([c[0] for c in confs], [idx] * len(confs))
        shift, sigma, _ = st.boltzmann(preds, [c[1] for c in confs])
        table = st.table("", smiles, idx, shift, sigma)
        groups = equivalence_groups(confs[0][0], idx)  # lowest-energy conformer
        merged = merge_table(table, groups, st.confidence)
    except Exception as exc:
        os.dup2(saved_stderr, 2)
        print(f"Error for {smiles!r}: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        os.dup2(saved_stderr, 2)
        os.close(devnull)

    cols = ["atom_index_1", "shift_ppm", "sigma_ppm", "confidence"]
    report = (
        f"CASCADE-2.0 13C prediction (Exp22K-FF-GPR, Boltzmann-weighted MMFF ensemble)\n"
        f"Date: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"SMILES: {smiles}\n"
        f"Conformers used: {len(confs)}\n"
        f"Atom numbers are 1-based positions in the SMILES.\n\n"
        f"Per carbon environment ({len(merged)} expected signals; equivalent carbons averaged,\n"
        f"diastereotopic carbons kept separate):\n"
        f"{merged.to_string(index=False)}\n\n"
        f"Per carbon atom ({len(table)} carbons):\n"
        f"{table[cols].to_string(index=False)}\n"
    )
    print(report)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
