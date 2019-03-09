#!/usr/bin/env python2
import horton
import numpy as np
import pandas as pd
import sys

def build_base(target, mixing, basisset):
    read = horton.io.xyz.load_xyz('benzene.xyz')
    
    target = np.array(target)
    baseline = np.array([7., 7., 7., 7., 7., 7, 0., 0., 0., 0., 0., 0.])
    dZ = target - baseline
    
    horton.log.set_level(0)
    mol = horton.IOData()
    mol.coordinates = read['coordinates']
    mol.numbers = baseline
    mol.pseudo_numbers = baseline + dZ * mixing
    print (mol.pseudo_numbers)
    
    # get grid
    grid = horton.BeckeMolGrid(mol.coordinates, (baseline * 0 + 7).astype(int), baseline * 0 + 7, 'insane', mode='keep', random_rotate=False)
    
    # build basis set
    basisset = basisset
    obasis = horton.get_gobasis(mol.coordinates[:6], baseline[:6].astype(int), basisset)
    for idx, dval in enumerate(dZ):
        if dval == 0:
            continue
        obasis2 = horton.get_gobasis(mol.coordinates[idx:idx+1], target[idx:idx+1].astype(int), basisset)
        obasis = horton.GOBasis.concatenate(obasis, obasis2)
    
    # start calculation
    olp = obasis.compute_overlap()
    kin = obasis.compute_kinetic()
    na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
    er = obasis.compute_electron_repulsion()
    
    orb_alpha = horton.Orbitals(obasis.nbasis)
    orb_beta = horton.Orbitals(obasis.nbasis)
    one = kin + na
    horton.guess_core_hamiltonian(olp, one, orb_alpha, orb_beta)
    
    external = {'nn': horton.compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    
    terms = [horton.UTwoIndexTerm(kin, 'kin'), horton.UDirectTerm(er, 'hartree'), horton.UExchangeTerm(er, 'x_hf'), horton.UTwoIndexTerm(na, 'ne')]
    ham = horton.UEffHam(terms, external)
    occ_model = horton.AufbauOccModel(21, 21)
    occ_model.assign(orb_alpha, orb_beta)
    dm_alpha = orb_alpha.to_dm()
    dm_beta = orb_beta.to_dm()
    scf_solver = horton.EDIIS2SCFSolver(1e-5, maxiter=400)
    scf_solver(ham, olp, occ_model, dm_alpha, dm_beta)
    
    fock_alpha = np.zeros(olp.shape)
    fock_beta = np.zeros(olp.shape)
    ham.reset(dm_alpha, dm_beta)
    energy = ham.compute_energy()
    ham.compute_fock(fock_alpha, fock_beta)
    orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, olp)
    orb_beta.from_fock_and_dm(fock_beta, dm_beta, olp)

    # integration grid
    rho_alpha = obasis.compute_grid_density_dm(dm_alpha, grid.points)
    rho_beta = obasis.compute_grid_density_dm(dm_beta, grid.points)
    rho_full = rho_alpha + rho_beta
    
    return energy, rho_full, external['nn']

t = map(int, sys.argv[1].split('-'))
res = build_base(t, float(sys.argv[2]), sys.argv[3])
fn = 'benzene' + '.'.join(sys.argv[1:]) + '.pkl.gz'
df = pd.DataFrame({'target': sys.argv[1], 'mixing': float(sys.argv[2]), 'basisset': sys.argv[3], 'energy': res[0], 'density': res[1], 'Enn': res[2]})
df.to_pickle(fn)
