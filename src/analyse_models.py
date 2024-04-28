#!/bin/python
#
# Copyright 2024 Yogesh Kalakoti, Linkoping University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import glob
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm

class EnsembleAnalyzer:
    def __init__(self, afout_path, pdb_state1, pdb_state2, outpath='results/'):
        self.afout_path = afout_path
        self.pdb_state1 = pdb_state1
        self.pdb_state2 = pdb_state2
        self.outpath = outpath
        self.state1_id = self.pdb_state1.split('/')[-1].split('.')[0]
        self.state2_id = self.pdb_state2.split('/')[-1].split('.')[0]

    def read_tmout(self, f):
        with open(f, "r") as file:
            for line in file.readlines():
                if 'user-specified d0' in line:    # TM-score
                    tm_score = float(line.split(' ')[1])
                if 'RMSD' in line:
                    try:
                        rmsd = float(line.split(' ')[7][:-1])
                    except:
                        rmsd = float(line.split(' ')[8][:-1])
        return tm_score, rmsd

    def calculate_mean_bfactor(self, pdb_file):
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

    def align_models(self, reference_pdb):
        print(f"Aligning models with {reference_pdb} as reference...")
        state_id = reference_pdb.split('/')[-1].split('.')[0]
        cmd = f"for pdb in {self.afout_path}/*.pdb;do echo TMalign $pdb {reference_pdb} -d 3.5'>'$pdb.{state_id}.TM; done | parallel --eta -j 64"
        os.system(cmd)

    def analyze_models(self):
        print("Analyzing models...")
        confidences = []
        for protein in tqdm(glob.glob(f"{self.afout_path}/*.pdb")):
            mean_bfactor = self.calculate_mean_bfactor(protein)
            confidences.append([protein, mean_bfactor])
        confidence_df = pd.DataFrame(confidences)

        top_confident = confidence_df.sort_values(by=1, ascending=False).iloc[0]
        top_confident_model = top_confident[0]
        confidence = top_confident[1]
        print(f"Most confident model: {top_confident_model}, Confidence: {confidence}")

        self.align_models(top_confident_model)
        self.align_models(self.pdb_state1)
        self.align_models(self.pdb_state2)

        print("Processing alignment results...")
        tm_with_best, tm_f1, tm_f2 = [], [], []
        for model in confidence_df[0]:
            tmout_file = f"{model}.{self.state1_id}.TM"
            tm, _ = self.read_tmout(tmout_file)
            tm_f2.append(tm)

            tmout_file = f"{model}.{self.state2_id}.TM"
            tm, _ = self.read_tmout(tmout_file)
            tm_f1.append(tm)

            tmout_file = f"{model}.bestmodel.TM"
            tm, _ = self.read_tmout(tmout_file)
            tm_with_best.append(tm)

        confidence_df['tm_f1'] = tm_f1
        confidence_df['tm_f2'] = tm_f2
        confidence_df['tm_with_best'] = tm_with_best
        confidence_df.to_csv(f'{self.outpath}/final_df_{self.state1_id}-{self.state2_id}.csv')
        print('\n>> Results csv saved at', f'{self.outpath}final_df_{self.state1_id}-{self.state2_id}.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse generated model ensemble')
    parser.add_argument('--afout_path', required=True, help='Path to generated models')
    parser.add_argument('--pdb_state1', required=True, help='Reference PDB of state1')
    parser.add_argument('--pdb_state2', required=True, help='Reference PDB of state2')
    parser.add_argument('--outpath', default='results/', help='Output path')

    args = parser.parse_args()

    model_analyzer = EnsembleAnalyzer(args.afout_path, args.pdb_state1, args.pdb_state2, args.outpath)
    model_analyzer.analyze_models()