"""Smoke test for running CASCADE-2.0 (13C, Exp22K-FF-GPR) locally.

Modes
  notebook    reproduce models/Predict_SMILES_FF_GPR/predictions_SMILES.ipynb
  extended    a few more molecules, one per batch, checked against approx. literature values
  conformers  single-conformer (several seeds) vs. MMFF conformer ensemble (Boltzmann)

Run with the Python 3.10 venv:
  .venv/bin/python smoke_test.py --mode notebook
"""
import argparse
import contextlib
import io
import math
import os
import pickle
import sys
import time
import warnings

# Never write .pyc files: CASCADE-2.0 tracks its __pycache__ in git, so imports would dirty the submodule.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)
MODEL_DIR = os.path.join(WORKSPACE, "CASCADE-2.0", "models", "Predict_SMILES_FF_GPR")
RESULTS_DIR = os.path.join(HERE, "results")

# model.py loads inducing_index_points_250.npy from the cwd at import time,
# and the code imports from modules/ (nfp, pooling, ...).
os.chdir(MODEL_DIR)
sys.path.insert(0, os.path.join(MODEL_DIR, "modules"))
sys.path.insert(0, MODEL_DIR)
sys.path.insert(0, HERE)  # conformers.py

from conformers import genConf  # noqa: E402  (MMFF conformer search, ported from CASCADE v1)

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")

from nfp.preprocessing import GraphSequence  # noqa: E402  (vendored in modules/)
from model import make_model  # noqa: E402

# Output scaling and confidence thresholds, from predictions_SMILES.ipynb
SHIFT_SCALE, SHIFT_OFFSET = 50.484337, 99.798111
CONF_LOW, CONF_MODERATE = 1.485774, 1.419831
ALLOWED_ELEMENTS = {1, 6, 7, 8, 9, 15, 16, 17}  # H C N O F P S Cl (server scope)
KT_KCAL = 0.001987 * 298.15  # as in CASCADE v1 apply.py


# --- needed by preprocessor_orig.p, which pickles __main__.atomic_number_tokenizer ---
def atomic_number_tokenizer(atom):
    return atom.GetAtomicNum()


sys.modules["__main__"].atomic_number_tokenizer = atomic_number_tokenizer


# --- copied from the notebook -------------------------------------------------------
def _compute_stacked_offsets(sizes, repeats):
    return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)


def ragged_const(inp_arr):
    return tf.ragged.constant(np.expand_dims(inp_arr, axis=0), ragged_rank=1)


def Mol_iter(df):
    for _, r in df.iterrows():
        yield (r["Mol"], r["atom_index"])


class RBFSequence(GraphSequence):
    def process_data(self, batch_data):
        offset = _compute_stacked_offsets(batch_data["n_pro"], batch_data["n_atom"])
        offset = np.where(batch_data["atom_index"] >= 0, offset, 0)
        batch_data["atom_index"] += offset

        features = ["node_attributes", "node_coordinates", "edge_indices", "atom_index", "n_pro"]
        for feature in features:
            batch_data[feature] = ragged_const(batch_data[feature])

        del batch_data["n_atom"]
        del batch_data["n_bond"]
        del batch_data["distance"]
        del batch_data["bond"]
        del batch_data["node_graph_indices"]
        return batch_data


# --- model ------------------------------------------------------------------------------
class Cascade2:
    def __init__(self):
        with open("preprocessor_orig.p", "rb") as f:
            self.preprocessor = pickle.load(f)["preprocessor"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = make_model()
        self.model.load_weights("best_model_val_mae.h5")

    def predict(self, mols, atom_indices, batch_size=32):
        """Return a list (one per mol) of (shift_ppm, sigma_ppm) arrays."""
        df = pd.DataFrame({"Mol": mols, "atom_index": atom_indices})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            inputs = self.preprocessor.predict(Mol_iter(df))
        seq = RBFSequence(inputs, batch_size=batch_size, shuffle=False)

        means, stds = [], []
        for b in range(len(seq)):
            batch_idx = [i for i in range(b * batch_size, (b + 1) * batch_size) if i < len(inputs)]
            expected = sum(int(inputs[i]["n_pro"]) for i in batch_idx)
            dist = self.model(seq[b])  # one call: take mean and stddev from the same distribution
            m = dist.mean().numpy().flatten()
            s = dist.stddev().numpy().flatten()
            if len(m) == expected + 1 and expected == 1:
                # modules/repeat.py duplicates the output for a single-molecule batch with one target atom
                m, s = m[:1], s[:1]
            if len(m) != expected:
                raise RuntimeError(f"batch {b}: got {len(m)} outputs, expected {expected}")
            means.append(m)
            stds.append(s)
        means, stds = np.concatenate(means), np.concatenate(stds)

        out, pos = [], 0
        for idx in atom_indices:
            n = len(idx)
            out.append((means[pos:pos + n] * SHIFT_SCALE + SHIFT_OFFSET, stds[pos:pos + n] * SHIFT_SCALE))
            pos += n
        return out


def confidence(sigma_ppm):
    x = 1.96 * sigma_ppm
    if x >= CONF_LOW:
        return "Low"
    if x >= CONF_MODERATE:
        return "Moderate"
    return "High"


# --- structures ---------------------------------------------------------------------------
def validate(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid SMILES"
    if len(Chem.GetMolFrags(mol)) > 1:
        return None, "more than one fragment"
    for a in mol.GetAtoms():
        if a.GetAtomicNum() not in ALLOWED_ELEMENTS:
            return None, f"element {a.GetSymbol()} outside model scope (H C N O F P S Cl)"
        if a.GetFormalCharge() != 0:
            return None, "charged atom"
    if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        return None, "no carbon atoms"
    return mol, None


def carbon_indices(mol):
    return np.array([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6], dtype=int)


def embed_single(mol, seed):
    """Notebook / training-data approach: one embedded, MMFF-optimised geometry."""
    m = Chem.AddHs(Chem.Mol(mol))
    if AllChem.EmbedMolecule(m, randomSeed=seed) != 0:
        raise RuntimeError("embedding failed")
    AllChem.MMFFOptimizeMolecule(m)
    return m


def embed_ensemble(mol, seed=-1):
    """CASCADE v1 web-app approach: 200 MMFF94s conformers, 10 kcal/mol window, 0.5 A RMSD prune.
    Returns list of (conformer_mol, relative_E_kcal). seed=-1 means random (as in v1)."""
    m = Chem.AddHs(Chem.Mol(mol))
    o, ids, _ = genConf(m, rms=-1, nc=200, efilter=10.0, rmspost=0.5, seed=seed)
    confs = []
    for e_rel, cid in ids:
        mi = Chem.Mol(o)
        mi.RemoveAllConformers()
        mi.AddConformer(Chem.Conformer(o.GetConformer(int(cid))), assignId=True)
        confs.append((mi, float(e_rel)))
    return confs


def boltzmann(results, energies):
    w = np.array([math.exp(-e / KT_KCAL) for e in energies])
    w /= w.sum()
    shifts = np.array([r[0] for r in results])
    sigmas = np.array([r[1] for r in results])
    return (w[:, None] * shifts).sum(0), (w[:, None] * sigmas).sum(0), w


def table(name, smiles, idx, shifts, sigmas):
    return pd.DataFrame({
        "name": name, "smiles": smiles, "atom_index": idx, "atom_index_1": idx + 1,
        "shift_ppm": np.round(shifts, 2), "sigma_ppm": np.round(sigmas, 3),
        "confidence": [confidence(s) for s in sigmas],
    })


# --- modes ------------------------------------------------------------------------------
def mode_notebook(model, seed):
    print("== Step 3: reproduce predictions_SMILES.ipynb ==")
    reference = {"CCO": [18.73, 58.63], "CC(=O)O": [20.96, 174.26], "c1ccccc1": [128.51] * 6}
    smiles_list = list(reference)
    mols = [embed_single(Chem.MolFromSmiles(s), seed) for s in smiles_list]
    idx = [carbon_indices(m) for m in mols]
    preds = model.predict(mols, idx, batch_size=32)  # all three in one batch, as in the notebook
    ok = True
    frames = []
    for s, i, (sh, sg) in zip(smiles_list, idx, preds):
        t = table(s, s, i, sh, sg)
        t["notebook_ref"] = reference[s]
        t["diff"] = (t.shift_ppm - t.notebook_ref).round(2)
        frames.append(t)
        ok &= bool((t["diff"].abs() <= 0.3).all())
    df = pd.concat(frames, ignore_index=True)
    print(df.drop(columns=["name", "atom_index_1"]).to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "smoke_notebook.csv"), index=False)
    print("\nRESULT:", "PASS (all within 0.3 ppm of notebook)" if ok else "CHECK (some > 0.3 ppm from notebook)")
    return ok


# approximate literature 13C values (CDCl3), compared as sorted lists
EXTENDED = [
    ("methanol", "CO", [50.4]),
    ("thymol", "Cc1ccc(C(C)C)c(O)c1", [152.6, 136.7, 131.5, 126.3, 121.8, 116.1, 26.8, 22.8, 22.8, 20.9]),
    ("(-)-menthol", "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O", [71.5, 50.1, 45.0, 34.5, 31.6, 25.8, 23.1, 22.2, 21.0, 16.1]),
    ("ethyl acetate", "CCOC(C)=O", [171.1, 60.4, 21.0, 14.2]),
    ("bromobenzene", "Brc1ccccc1", None),
]


def mode_extended(model, seed):
    print("== Step 4: extended molecules (one molecule per batch) ==")
    frames, summary, ok = [], [], True
    for name, smi, lit in EXTENDED:
        mol, err = validate(smi)
        if err:
            print(f"\n{name:14s} {smi}: REJECTED - {err}")
            summary.append((name, "rejected", err))
            if lit is not None:
                ok = False
            continue
        m = embed_single(mol, seed)
        idx = carbon_indices(m)
        (sh, sg), = model.predict([m], [idx], batch_size=1)
        if len(sh) != len(idx):
            ok = False
        t = table(name, smi, idx, sh, sg)
        frames.append(t)
        print(f"\n{name}  {smi}")
        print(t[["atom_index", "shift_ppm", "sigma_ppm", "confidence"]].to_string(index=False))
        if lit is not None:
            mae = float(np.mean(np.abs(np.sort(sh) - np.sort(lit))))
            print(f"  sorted-list MAE vs approx. literature: {mae:.2f} ppm  ({len(sh)} C, expected {len(lit)})")
            summary.append((name, f"{len(sh)} C", f"MAE {mae:.2f} ppm"))
            ok &= (len(sh) == len(lit)) and mae < 2.5
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(RESULTS_DIR, "smoke_extended.csv"), index=False)
    print("\nSummary:")
    for row in summary:
        print("  {:14s} {:10s} {}".format(*row))
    print("\nRESULT:", "PASS" if ok else "CHECK")
    return ok


def mode_conformers(model, seeds):
    print("== Step 5: single conformer (several seeds) vs MMFF ensemble ==")
    rows = []
    for name, smi, lit in EXTENDED[1:3]:  # thymol, menthol
        mol, _ = validate(smi)
        idx = carbon_indices(Chem.AddHs(mol))

        t0 = time.time()
        singles = [embed_single(mol, s) for s in seeds]
        t_embed_single = (time.time() - t0) / len(seeds)
        t0 = time.time()
        single_preds = model.predict(singles, [idx] * len(singles))
        t_pred_single = (time.time() - t0) / len(seeds)
        single_shifts = np.array([p[0] for p in single_preds])

        t0 = time.time()
        confs = embed_ensemble(mol)
        t_embed_ens = time.time() - t0
        t0 = time.time()
        ens_preds = model.predict([c[0] for c in confs], [idx] * len(confs))
        t_pred_ens = time.time() - t0
        ens_shift, ens_sigma, w = boltzmann(ens_preds, [c[1] for c in confs])

        df = pd.DataFrame({
            "atom_index": idx,
            **{f"seed{s}": np.round(single_shifts[k], 2) for k, s in enumerate(seeds)},
            "single_mean": np.round(single_shifts.mean(0), 2),
            "seed_spread": np.round(single_shifts.max(0) - single_shifts.min(0), 2),
            "ensemble": np.round(ens_shift, 2),
            "ens_minus_single": np.round(ens_shift - single_shifts.mean(0), 2),
        })
        print(f"\n{name}  {smi}   ensemble: {len(confs)} conformers kept, "
              f"top Boltzmann weights {np.round(np.sort(w)[::-1][:3], 3).tolist()}")
        print(df.to_string(index=False))
        mae_single = float(np.mean(np.abs(np.sort(single_shifts.mean(0)) - np.sort(lit))))
        mae_ens = float(np.mean(np.abs(np.sort(ens_shift) - np.sort(lit))))
        print(f"  max seed spread {df.seed_spread.max():.2f} ppm | max |ensemble - single| "
              f"{df.ens_minus_single.abs().max():.2f} ppm")
        print(f"  sorted MAE vs approx. literature: single(mean of seeds) {mae_single:.2f}, ensemble {mae_ens:.2f}")
        print(f"  time per molecule: single {t_embed_single + t_pred_single:.2f} s "
              f"(embed {t_embed_single:.2f}, predict {t_pred_single:.2f}); ensemble "
              f"{t_embed_ens + t_pred_ens:.2f} s (conformers {t_embed_ens:.2f}, predict {t_pred_ens:.2f})")
        df.insert(0, "name", name)
        rows.append(df)
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(RESULTS_DIR, "smoke_conformers.csv"), index=False)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["notebook", "extended", "conformers", "all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t0 = time.time()
    model = Cascade2()
    print(f"model loaded in {time.time() - t0:.1f} s\n")
    if args.mode in ("notebook", "all"):
        mode_notebook(model, args.seed)
        print()
    if args.mode in ("extended", "all"):
        mode_extended(model, args.seed)
        print()
    if args.mode in ("conformers", "all"):
        mode_conformers(model, seeds=[1, 2, 3, 4, 5])


if __name__ == "__main__":
    main()
