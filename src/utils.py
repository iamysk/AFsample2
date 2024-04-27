import pandas as pd
import glob

# Extract bfactors from pdb
from Bio.PDB import PDBParser
from tqdm import tqdm

def read_tmout(f):
    with open(f, "r") as file:
        for line in file.readlines():
            if 'user-specified d0' in line:    # TM-score
                tm_score = float(line.split(' ')[1])
            if 'RMSD' in line:
                #print(line.split(' '))
                try:
                    rmsd=float(line.split(' ')[7][:-1])
                except:
                    rmsd=float(line.split(' ')[8][:-1]) 
    file.close()
    return tm_score, rmsd


def calculate_mean_bfactor(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('pdb_structure', pdb_file)
    
    total_bfactor = 0.0
    num_atoms = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    total_bfactor += atom.bfactor
                    num_atoms += 1
    
    if num_atoms == 0:
        return None
    
    mean_bfactor = total_bfactor / num_atoms
    return mean_bfactor