#!/usr/bin/env python

import numpy as np
import pandas as pd
import horton
import multiprocessing as mp

def get_grid(bond_length):
    coordinates = np.array([[0.0, 0.0, -bond_length/2.], [0.0, 0.0, bond_length/2.]])
    grid = horton.BeckeMolGrid(coordinates, np.array([7, 7]), np.array([7.,7.]), 'insane', mode='keep', random_rotate=False)
    return grid, coordinates

def build_grids():
    DEBUG = False
    grids = []
    if DEBUG:
        bls = [1, 2]
    else:
        bls = np.hstack((np.linspace(1, 2.5, 11), np.linspace(3, 5, 5), [10., 20., 30.] ))
    for bond_length in bls:
        grid, coordinates = get_grid(bond_length)
        grids.append({'bond': bond_length, 'points': grid.points, 'weights': grid.weights, 'coordinates': coordinates})

    return pd.DataFrame(grids)

grids = build_grids()
grids.to_pickle('N2.pkl')

