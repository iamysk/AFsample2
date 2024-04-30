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
import re
import collections, operator
import glob
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm

class EnsembleAnalyzer:
    def __init__(self, jobid, afout_path, outpath='results/', pdb_state1=None, pdb_state2=None):
        self.jobid = jobid
        self.afout_path = afout_path
        self.pdb_state1 = pdb_state1
        self.pdb_state2 = pdb_state2
        self.outpath = outpath
        self.get_state = GetState(jobid, radius=1)     # Initialize GetState class instance

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

    def calculate_bfactors(self, pdb_file):
        parser = PDBParser()
        structure = parser.get_structure('pdb_structure', pdb_file)

        per_res_bfactors = []
        num_atoms = 0

        for model in structure:
            for chain in model:
                for residue in chain:
                    per_res_bfactors.append(np.sum([atom.bfactor for atom in residue])/len(residue))

        return per_res_bfactors

    def align_models(self, reference_pdb, reference=False):
        print(f"Aligning models with {reference_pdb}")
        #if reference:
        state_id = reference_pdb.split('/')[-1].split('.')[0]
        cmd = f"for pdb in {self.afout_path}/*.pdb;do echo TMalign $pdb {reference_pdb} -d 3.5'>'$pdb.{state_id}.TM; done | parallel --eta -j 64"
        os.system(cmd)
        return state_id
    
    def make_tm_df(self, df, mode):
        tms = []
        for model in df['model_path']:
            tmout_file = f"{model}.{mode}.TM"
            tm, _ = self.read_tmout(tmout_file)
            tms.append(tm)
        df['tm_'+mode] = tms
        return df

    def analyze_models(self):
        print("Analyzing models...")
        print('Reference state1:', self.pdb_state1)
        print('Reference state2:', self.pdb_state2)

        overall_confidences, per_residue_confidences = [], []
        for protein in tqdm(glob.glob(f"{self.afout_path}/*.pdb")):
            bfactors = self.calculate_bfactors(protein)
            per_residue_confidences.append(bfactors)
            overall_confidences.append([protein, np.mean(bfactors)])
        confidence_df = pd.DataFrame(overall_confidences, columns=['model_path', 'confidence'])
        lowconf_indices = np.where(np.mean(per_residue_confidences, axis=0)<50)[0]+1  # Get indices of low-confidence residues. Rosetta is 1-indexed
        print('Low confidence indices:', lowconf_indices)
        
        # Get top confident model
        top_confident = confidence_df.sort_values(by='confidence', ascending=False).iloc[0]
        top_confident_model = top_confident['model_path']
        confidence = top_confident['confidence']
        print(f"Most confident model: {top_confident_model}, Confidence: {confidence}")

        # Get states from geenrated models if not provided
        if self.pdb_state1 is None or self.pdb_state2 is None:
            reference=False
            print('\n========== Running TM-align ==========')
            bestmodel = self.align_models(top_confident_model, reference=reference)
            confidence_df = self.make_tm_df(confidence_df, bestmodel)
            print(confidence_df.head())
            self.pdb_state1, self.pdb_state2 = self.get_state.calculate_states(confidence_df, lowconf_indices)
        else:
            reference=True
            print('>> Received reference pdbs...', self.pdb_state1, self.pdb_state2)

        print('\n========== Running TM-align ==========')

        if reference:
            bestmodel = self.align_models(top_confident_model)
            confidence_df = self.make_tm_df(confidence_df, top_confident_model)

        state1_id = self.align_models(self.pdb_state1, reference=reference)
        state2_id = self.align_models(self.pdb_state2, reference=reference)
        print('>> Alignents done. TM-align outputs saved at', self.afout_path)

        print(">> Processing alignment results...")
        print(state1_id)
        confidence_df = self.make_tm_df(confidence_df, state1_id)
        confidence_df = self.make_tm_df(confidence_df, state2_id)
        
        outfile = f'{self.outpath}/final_df_{self.jobid}_{state1_id}-{state2_id}.csv'
        confidence_df.to_csv(outfile)
        print('\n>> Results csv saved at', outfile)

class GetState():
    def __init__(self, jobid, radius):
        self.jobid = jobid
        self.radius = radius

    def read_clusterfile(self, clusterfile):
        data=[]
        with open(clusterfile) as f:
            for line in f:
                match=re.match('protocols.cluster: \(0\)\s+\d+\s+(\S+)\s+(\d+)\s+\d+',line)
                if match:
                    row={}
                    row['model']=match.group(1)             #line.split()[3]
                    row['cluster']=int(match.group(2))      #int(line.split()[4])
                    data.append(row)
        return(pd.DataFrame(data))
    
    def calculate_states(self, confidence_df, lowconf_indices):
        tempdir = f'temp/{self.jobid}'
        if not os.path.exists(tempdir):
            os.makedirs(tempdir)

        # Make input file for rosetta
        with open(f'{tempdir}/{self.jobid}.txt', 'w') as fp:
            for pdb in confidence_df['model_path']:
                fp.write(f"{pdb}\n")
        fp.close()
        
        rosetta_path = '/software/presto/software/Rosetta/3.13-foss-2019b-2/bin/cluster.mpi.linuxgccrelease'
        outfile = f'{tempdir}/model_all_{self.radius}_cluster.out'
        if os.path.isfile(outfile):
            pass
        else:
            if len(lowconf_indices)>0:
                cmd = f'{rosetta_path} -in:file:l {tempdir}/{self.jobid}.txt -in:file:fullatom -score:empty -out:prefix {tempdir}/model_all_{self.radius}_ -cluster:radius {self.radius} -cluster:exclude_res {" ".join(map(str, lowconf_indices))} >{outfile}'
            else:
                cmd = f'{rosetta_path} -in:file:l {tempdir}/{self.jobid}.txt -in:file:fullatom -score:empty -out:prefix {tempdir}/model_all_{self.radius}_ -cluster:radius {self.radius} >{outfile}'
            print(cmd)
            os.system(cmd)

        df_cluster = self.read_clusterfile(outfile)
        counter_dict = collections.Counter(df_cluster['cluster'].values)

        tm_w_best_col = confidence_df.columns[-1]
        confidence_df['cluster_ids']=df_cluster['cluster']
        
        # Sorting by values
        sorted_by_values = {key: value for key, value in sorted(counter_dict.items(), key=lambda item: item[1], reverse=True)}
        rand_df_clusters = confidence_df[confidence_df['cluster_ids'].isin(list(sorted_by_values.keys())[:5])]

        # Filter by confidence
        subset_filtered1 = rand_df_clusters[rand_df_clusters['confidence']>=0.60].reset_index()
        print(subset_filtered1.shape)
        
        xmax, xmin = subset_filtered1[tm_w_best_col].min(), subset_filtered1[tm_w_best_col].max()
        print(xmax, xmin)

        max_indices = subset_filtered1.groupby('cluster_ids')['confidence'].idxmax()    # best model from each cluster by confidence
        best_models_for_each_cluster = subset_filtered1.loc[max_indices].sort_values(by=tm_w_best_col, ascending=False).reset_index(drop=True)     # best model from each cluster by confidence
        
        print(best_models_for_each_cluster)
        _, pdb_state1, tm_w_best1, confidence1, clusterid1 = list(best_models_for_each_cluster.values[0])
        _, pdb_state2, tm_w_best2, confidence2, clusterid2 =list(best_models_for_each_cluster.values[-1])

        print('\n=========== Summary of state identification ===========')
        print('Total models analysed:', len(confidence_df))
        print('Number of clusters:', len(counter_dict))
        print('State, model_pdb, tm_w_best, confidence, clusterid')
        print(f"1, {pdb_state1}, {tm_w_best1}, {confidence1}, {clusterid1}")
        print(f"2, {pdb_state2}, {tm_w_best2}, {confidence2}, {clusterid2}")
        return pdb_state1, pdb_state2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse generated model ensemble')
    parser.add_argument('--jobid', required=True, help='jobid')
    parser.add_argument('--afout_path', required=True, help='Path to generated models')
    parser.add_argument('--pdb_state1', required=False, help='Reference PDB of state1')
    parser.add_argument('--pdb_state2', required=False, help='Reference PDB of state2')
    parser.add_argument('--outpath', default='results/', help='Output path')

    args = parser.parse_args()

    model_analyzer = EnsembleAnalyzer(args.jobid, args.afout_path, args.outpath, args.pdb_state1, args.pdb_state2)
    model_analyzer.analyze_models()