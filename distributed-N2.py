#!/usr/bin/env python

import numpy as np
import pandas as pd
import horton
import multiprocessing as mp

def get_hf_density(bond_length, mixing_value, dZ):
    horton.log.set_level(0)

    mol = horton.IOData()
    bond_length = float(bond_length)
    dZ = float(dZ)
    mol.coordinates = np.array([[0.0, 0.0, -bond_length/2.], [0.0, 0.0, bond_length/2.]])
    mol.numbers = np.array([7., 7.])
    mol.pseudo_numbers = np.array([7., 7.]) + dZ * np.array([-1., 1.]) * mixing_value
    
    grid = horton.BeckeMolGrid(mol.coordinates, np.array([7, 7]), np.array([7.,7.]), 'insane', mode='keep', random_rotate=False)

    # build basis set
    basisset = 'STO-3G' #'6-31G(d)'
    obasis = horton.get_gobasis(mol.coordinates, np.array([7, 7]), basisset)
    obasis2 = horton.get_gobasis(mol.coordinates, np.array([7-int(dZ), 7+int(dZ)]), basisset)
    obasis.concatenate(obasis2)

    lf = horton.DenseLinalgFactory(obasis.nbasis)
    olp = obasis.compute_overlap(lf)
    kin = obasis.compute_kinetic(lf)
    na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, lf)
    er = obasis.compute_electron_repulsion(lf)
    
    #orb_alpha = horton.Orbitals(obasis.nbasis)
    #orb_beta = horton.Orbitals(obasis.nbasis)
    orb_alpha = lf.create_expansion()
    orb_beta = lf.create_expansion()
    
    horton.guess_core_hamiltonian(olp, kin, na, orb_alpha, orb_beta)
    
    external = {'nn': horton.compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    
    terms = [horton.UTwoIndexTerm(kin, 'kin'), horton.UDirectTerm(er, 'hartree'), horton.UExchangeTerm(er, 'x_hf'), horton.UTwoIndexTerm(na, 'ne')]
    ham = horton.UEffHam(terms, external)
    occ_model = horton.AufbauOccModel(7, 7)
    occ_model.assign(orb_alpha, orb_beta)
    dm_alpha = orb_alpha.to_dm()
    dm_beta = orb_beta.to_dm()
    scf_solver = horton.ODASCFSolver(1e-5, maxiter=400)
    #scf_solver = horton.PlainSCFSolver(1e-6)
    scf_solver(ham, lf, olp, occ_model, dm_alpha, dm_beta)
    
    #fock_alpha = np.zeros(olp.shape)
    #fock_beta = np.zeros(olp.shape)
    ham.reset(dm_alpha, dm_beta)
    energy = ham.compute_energy()
    #ham.compute_fock(fock_alpha, fock_beta)
    #orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, olp)
    #orb_beta.from_fock_and_dm(fock_beta, dm_beta, olp)

    # integration grid
    rho_alpha = obasis.compute_grid_density_dm(dm_alpha, grid.points)
    rho_beta = obasis.compute_grid_density_dm(dm_beta, grid.points)
    rho_full = rho_alpha + rho_beta
    
    ener, den, nuc = energy, rho_full, external['nn']
    return {'bond': bond_length, 'mixing': mixing_value, 'energy': ener, 'density': den, 'Enn': nuc, 'dZ': dZ}

def worker(q, results):
    import numpy as np
    import pandas as pd
    import horton
    while True:
        message = q.get()
        if message == None:
            q.task_done()
            break
        print (message)
        try:
            ret = get_hf_density(*message)
            results.put(ret)
        except:
            pass
        q.task_done()

def build_cache(dZ=1):
    q = mp.JoinableQueue()
    results = mp.Queue()
    NUMWORKERS = 32
    DEBUG = True

    for w in range(NUMWORKERS):
        wk = mp.Process(target=worker, args=((q, results)))
        wk.daemon = True
        wk.start()

    densities = {}

    if DEBUG:
        bls = [1, 2]
    else:
        bls = np.hstack((np.linspace(1, 2.5, 11), np.linspace(3, 5, 5), [10., 20., 30.] ))
    for bond_length in bls:
        density_path = {}
        for mixing_value in np.linspace(0, 1, 17):
            #ret.append(get_hf_density(bond_length, mixing_value))
            q.put([bond_length, mixing_value, dZ])
        densities[bond_length] = density_path

    for w in range(NUMWORKERS):
        q.put(None)

    q.join()

    ret = []
    while results.qsize() != 0:
        ret.append(results.get())

    return pd.DataFrame(ret)

for dZ in (1, 2, 3, 4, 5, 6):
    cache = build_cache(dZ)
    cache.to_pickle('cache-%d.pkl' % dZ)

