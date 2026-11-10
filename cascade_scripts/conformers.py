"""MMFF conformer search used for the CASCADE ensemble.

Re-implementation of genConf / energy_filter / postrmsd from CASCADE v1
(patonlab/CASCADE, cascade-Jupyternotebook-SMILES/models/cascade/genConf.py, MIT licence).
Same algorithm and parameters; logging output and the dead fallback branch removed.

genConf(m, nc, rms, efilter, rmspost) -> (mol_with_conformers, [(E_rel_kcal, conf_id), ...], n_rotatable)
  1. embed nc conformers (default 3**n_rotatable if nc is falsy)
  2. MMFF94s-minimise each, record its energy
  3. keep conformers within efilter kcal/mol of the minimum
  4. drop conformers whose heavy-atom RMSD to an already-kept (lower-energy) one is < rmspost
"""
from rdkit import Chem
from rdkit.Chem import AllChem


def genConf(m, nc, rms, efilter, rmspost, seed=-1):
    nr = int(AllChem.CalcNumRotatableBonds(m))
    Chem.AssignAtomChiralTagsFromStructure(m, replaceExistingTags=True)
    if not nc:
        nc = 3 ** nr

    ids = list(AllChem.EmbedMultipleConfs(m, numConfs=nc, randomSeed=seed))
    if not ids:  # retry from random coordinates (hard-to-embed molecules)
        ids = list(AllChem.EmbedMultipleConfs(m, numConfs=nc, randomSeed=seed, useRandomCoords=True))
    if not ids:
        raise RuntimeError("cannot generate FF conformers")

    prop = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
    if prop is None:
        raise RuntimeError("MMFF94s parameters not available for this molecule")
    diz = []
    for cid in ids:
        ff = AllChem.MMFFGetMoleculeForceField(m, prop, confId=cid)
        ff.Minimize()
        diz.append((float(ff.CalcEnergy()), cid))

    n, diz2 = energy_filter(m, diz, efilter)

    if rmspost is not None and n.GetNumConformers() > 1:
        o, diz3 = postrmsd(n, diz2, rmspost)
    else:
        o, diz3 = n, diz2
    return o, diz3, nr


def energy_filter(m, diz, efilter):
    """Keep conformers within efilter kcal/mol of the lowest; energies returned relative to it."""
    diz = sorted(diz)
    mini = diz[0][0]
    n = Chem.Mol(m)
    n.RemoveAllConformers()
    kept = []
    for e, cid in diz:
        if e - mini > efilter:
            break
        n.AddConformer(m.GetConformer(int(cid)))  # keeps the original conformer id
        kept.append((e - mini, int(cid)))
    return n, kept


def postrmsd(n, diz2, rmspost):
    """Drop conformers within rmspost (heavy-atom best RMS) of an already-kept, lower-energy one."""
    diz2 = sorted(diz2, key=lambda x: x[0])
    o = Chem.Mol(n)
    nh = Chem.RemoveHs(n)
    confidlist = [diz2[0][1]]
    enval = [diz2[0][0]]
    for z, w in diz2[1:]:
        confid = int(w)
        if all(AllChem.GetBestRMS(nh, nh, prbId=confid, refId=ref) >= rmspost for ref in confidlist):
            confidlist.append(confid)
            enval.append(float(z))
    return o, list(zip(enval, confidlist))
