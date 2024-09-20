#!/usr/bin/env python3
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
import collections
import glob
import logging
import os
import re
import sys
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GetState:
    def __init__(self, jobid: str, radius: int):
        self.jobid = jobid
        self.radius = radius

    def read_clusterfile(self, clusterfile: str) -> pd.DataFrame:
        data = []
        pattern = re.compile(r'protocols\.cluster: \(0\)\s+\d+\s+(\S+)\s+(\d+)\s+\d+')
        try:
            with open(clusterfile, 'r') as f:
                for line in f:
                    match = pattern.match(line)
                    if match:
                        data.append({
                            'model': match.group(1),
                            'cluster': int(match.group(2))
                        })
        except FileNotFoundError:
            logger.error(f"Cluster file not found: {clusterfile}")
            raise
        except Exception as e:
            logger.error(f"Error reading cluster file {clusterfile}: {e}")
            raise

        return pd.DataFrame(data)

    def calculate_states(
        self,
        confidence_df: pd.DataFrame,
        lowconf_indices: np.ndarray
    ) -> Tuple[str, str, pd.DataFrame]:
        tempdir = Path(f'temp/{self.jobid}')
        tempdir.mkdir(parents=True, exist_ok=True)

        input_file = tempdir / f'{self.jobid}.txt'
        try:
            confidence_df['model_path'].to_csv(input_file, index=False, header=False)
            logger.info(f"Input file for Rosetta clustering created at {input_file}")
        except Exception as e:
            logger.error(f"Error writing input file for Rosetta: {e}")
            raise

        rosetta_path = '/software/presto/software/Rosetta/3.13-foss-2019b-2/bin/cluster.mpi.linuxgccrelease'
        outfile = tempdir / f'model_all_{self.radius}_cluster.out'

        if not outfile.is_file():
            if len(lowconf_indices) > 0:
                exclude_res = ' '.join(map(str, lowconf_indices))
                cmd = [
                    rosetta_path,
                    '-in:file:l', str(input_file),
                    '-in:file:fullatom',
                    '-score:empty',
                    '-out:prefix', str(tempdir / f'model_all_{self.radius}_'),
                    '-cluster:radius', str(self.radius),
                    '-cluster:exclude_res', exclude_res
                ]
            else:
                cmd = [
                    rosetta_path,
                    '-in:file:l', str(input_file),
                    '-in:file:fullatom',
                    '-score:empty',
                    '-out:prefix', str(tempdir / f'model_all_{self.radius}_'),
                    '-cluster:radius', str(self.radius)
                ]

            logger.info(f"Running Rosetta clustering with command: {' '.join(cmd)}")
            try:
                with open(outfile, 'w') as out_f:
                    subprocess.run(cmd, check=True, stdout=out_f, stderr=subprocess.PIPE)
                logger.info(f"Rosetta clustering completed. Output saved to {outfile}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Rosetta clustering failed: {e.stderr.decode().strip()}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during Rosetta clustering: {e}")
                raise
        else:
            logger.info(f"Cluster output file already exists: {outfile}")

        df_cluster = self.read_clusterfile(str(outfile))
        cluster_counts = collections.Counter(df_cluster['cluster'].values)
        sorted_clusters = sorted(cluster_counts.items(), key=lambda item: item[1], reverse=True)
        top_clusters = [cluster for cluster, _ in sorted_clusters[:5]]
        logger.info(f"Top 5 clusters: {top_clusters}")

        rand_df_clusters = confidence_df[confidence_df['cluster_ids'].isin(top_clusters)]
        logger.info(f"RAND DF SHAPE (TOP 5 CLUSTERS): {rand_df_clusters['cluster_ids'].unique()}")

        subset_filtered1 = rand_df_clusters[rand_df_clusters['confidence'] >= 0.60].reset_index(drop=True)
        logger.info(f"subset_filtered1.shape: {subset_filtered1.shape}")

        if subset_filtered1.empty:
            logger.warning("No models passed the confidence threshold of 0.60")
            raise ValueError("No models passed the confidence threshold of 0.60")

        max_indices = subset_filtered1.groupby('cluster_ids')['confidence'].idxmax()
        best_models = subset_filtered1.loc[max_indices].sort_values(by='tm_best', ascending=False).reset_index(drop=True)

        if best_models.empty or len(best_models) < 2:
            logger.error("Insufficient models after filtering for state identification")
            raise ValueError("Insufficient models after filtering for state identification")

        state1 = best_models.iloc[0]
        state2 = best_models.iloc[-1]

        logger.info("\n=========== Summary of state identification ===========")
        logger.info(f"Total models analysed: {len(confidence_df)}")
        logger.info(f"Number of clusters: {len(cluster_counts)}")
        logger.info("State, model_pdb, tm_best, confidence, clusterid")
        logger.info(f"1, {state1['model']}, {state1['tm_best']}, {state1['confidence']}, {state1['cluster_ids']}")
        logger.info(f"2, {state2['model']}, {state2['tm_best']}, {state2['confidence']}, {state2['cluster_ids']}")

        return state1['model'], state2['model'], rand_df_clusters


class EnsembleAnalyzer:
    def __init__(
        self,
        jobid: str,
        afout_path: str,
        clustering: bool,
        ncpu: int,
        outpath: str = 'results/',
        pdb_state1: Optional[str] = None,
        pdb_state2: Optional[str] = None,
        
    ):
        self.jobid = jobid
        self.afout_path = Path(afout_path)
        self.pdb_state1 = pdb_state1
        self.pdb_state2 = pdb_state2
        self.outpath = Path(outpath)
        self.clustering = clustering
        self.ncpu = ncpu
        self.get_state = GetState(jobid, radius=1)

    # def read_tmout(self, filepath: Path) -> Tuple[float, float]:
    #     tm_score = rmsd = 0.0
    #     try:
    #         with filepath.open('r') as file:
    #             for line in file:
    #                 if 'user-specified d0' in line:
    #                     tm_score = float(line.split()[1])
    #                 if 'RMSD' in line:
    #                     parts = line.split()
    #                     try:
    #                         rmsd = float(parts[7].rstrip(','))
    #                     except IndexError:
    #                         rmsd = float(parts[8].rstrip(','))
    #     except FileNotFoundError:
    #         logger.error(f"TM-out file not found: {filepath}")
    #         raise
    #     except Exception as e:
    #         logger.error(f"Error reading TM-out file {filepath}: {e}")
    #         raise

    #     return tm_score, rmsd

    def calculate_bfactors(self, pdb_file: Path) -> List[float]:
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure('pdb_structure', str(pdb_file))
        except Exception as e:
            logger.error(f"Error parsing PDB file {pdb_file}: {e}")
            raise

        per_res_bfactors = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if len(residue) == 0:
                        continue
                    b_factors = [atom.bfactor for atom in residue]
                    average_b = np.mean(b_factors)
                    per_res_bfactors.append(average_b)

        return per_res_bfactors

    
    def tmalign_and_extract_scores(self, pdb1, pdb2, tmalign_path='TMalign'):
        try:
            result = subprocess.run([tmalign_path, pdb1, pdb2, '-d', '3.5'], capture_output=True, text=True, check=True)
            # logging.info(f'TM-align between {pdb1} and {pdb2} successful.')
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running TMalign: {e}")
            return None, None

        output = result.stdout
        lines = output.splitlines()
        tm_score = rmsd = None
        for line in lines:
            if 'user-specified d0' in line:
                try:
                    line = line.split('=')[1]
                    line = line.split(' ')
                    tm_score = float(line[1])
                except (IndexError, ValueError):
                    logging.error(f"Error parsing TM-score from line: {line}")
            if 'RMSD' in line:
                try:
                    rmsd = float(line.split(' ')[-3].split(',')[0])
                except (IndexError, ValueError):
                    try:
                        rmsd = float(line.split(' ')[8][:-1])
                    except IndexError:
                        logging.error(f"Error parsing RMSD from line: {line}")

        return tm_score, rmsd
    
    def process_model(self, model: str, reference: str)-> dict:
        tm1, rm1 = self.tmalign_and_extract_scores(model, reference, tmalign_path='TMalign')
        return {'model': model, 'tm1': tm1, 'rm1': rm1}
    
    def run_tmalign(self, models: list, reference: str, mode: str)-> pd.DataFrame:
        args = [(model, reference) for model in models]

        with Pool(processes=self.ncpu) as pool:
            results = list(tqdm(pool.starmap(self.process_model, args), total=len(models), desc="Running TM-align"))

        df = pd.DataFrame(results)
        df = df.rename(columns={'tm1': f'tm_w_{mode}', 'rm1': f'rm_w_{mode}'})
        return df

    def make_tm_df(self, df: pd.DataFrame, mode: str, tmoutdir: Path) -> pd.DataFrame:
        tms = []
        for model in df['model']:
            tmout_file = tmoutdir / f"{Path(model).name}.{mode}.TM"
            tm, _ = self.read_tmout(tmout_file)
            tms.append(tm)
        df[f'tm_{mode}'] = tms
        logger.debug(f"Added TM scores for mode '{mode}' to DataFrame")
        return df

    def classify_samples(self, state1: Tuple, state2: Tuple) -> Tuple[int, int]:
        tm_s1_o, tm_s1_c, _ = state1
        tm_s2_o, tm_s2_c, _ = state2
        if tm_s1_o + tm_s2_c < tm_s1_c + tm_s2_o:
            return 1, 2
        else:
            return 2, 1

    def analyze_models(self):
        logger.info("Analyzing models...")
        logger.info(f'Reference state1: {self.pdb_state1}')
        logger.info(f'Reference state2: {self.pdb_state2}')

        overall_confidences = []
        per_residue_confidences = []

        models = list(self.afout_path.glob("unrelaxed*.pdb"))
        logger.info(f'Found {len(models)} models in {self.afout_path}')

        for protein in tqdm(models, desc="Reading confidence"):
            try:
                bfactors = self.calculate_bfactors(protein)
                per_residue_confidences.append(bfactors)
                overall_confidences.append([str(protein), np.mean(bfactors)])
            except Exception as e:
                logger.warning(f"Skipping model {protein} due to error: {e}")
                continue

        if not overall_confidences:
            logger.error("No models processed successfully.")
            raise ValueError("No models processed successfully.")

        confidence_df = pd.DataFrame(overall_confidences, columns=['model', 'confidence'])
        lowconf_indices = np.where(np.mean(per_residue_confidences, axis=0) < 50)[0] + 1  # 1-indexed
        logger.info(f'Low confidence residue indices: {lowconf_indices}')

        # Get top confident model
        top_confident_df = confidence_df.nlargest(1,'confidence')
        top_confident_model = Path(top_confident_df['model'].values[0])
        max_confidence = top_confident_df['confidence'].values[0]
        logger.info(f"Most confident model: {top_confident_model}, Confidence: {max_confidence}")

        tmoutdir = self.outpath / self.jobid / 'tmout'
        tmoutdir.mkdir(parents=True, exist_ok=True)

        if self.pdb_state1 is None or self.pdb_state2 is None:
            reference = False
            logger.info('========== Running TM-align with msot confident model ==========')
            tmdf = self.run_tmalign(models, top_confident_model, mode='best')
            confidence_df['model'] = confidence_df['model'].astype(str).str.strip()
            tmdf['model'] = tmdf['model'].astype(str).str.strip()

            merged_df = pd.merge(confidence_df, tmdf, on='model')            

            if self.clustering:
                logger.info('Initializing Rosetta clustering since clustering=True')
                self.pdb_state1, self.pdb_state2, filtered_df = self.get_state.calculate_states(
                    merged_df, lowconf_indices
                )
            else:
                conf_threshold = max_confidence * 0.60
                conf_filtered = merged_df[merged_df['confidence'] >= conf_threshold]
                logger.debug(f"Filtered models with confidence >= {conf_threshold}: {conf_filtered.shape}")

                if conf_filtered.empty:
                    logger.error("No models passed the confidence threshold.")
                    raise ValueError("No models passed the confidence threshold.")

                idx1 = conf_filtered['tm_w_best'].idxmax()
                state1 = conf_filtered.loc[idx1]

                idx2 = conf_filtered['tm_w_best'].idxmin()
                state2 = conf_filtered.loc[idx2]

                self.pdb_state1, self.pdb_state2 = state1['model'], state2['model']
                #logger.info(f'Identified states: {self.pdb_state1}, {self.pdb_state2}')
                filtered_df = conf_filtered
        else:
            reference = True
            logger.info(f'Received reference PDBs: {self.pdb_state1}, {self.pdb_state2}')

        logger.info('========== Running TM-align with identified states ==========')

        if reference:
            #bestmodel_id = self.align_models(top_confident_model, tmoutdir)
            #confidence_df = self.make_tm_df(confidence_df, bestmodel_id, tmoutdir)
            filtered_df = confidence_df

        tms1_df = self.run_tmalign(models, Path(self.pdb_state1), mode='s1')
        tms2_df = self.run_tmalign(models, Path(self.pdb_state2), mode='s2')

        tms1_df['model'] = tms1_df['model'].astype(str).str.strip()
        tms2_df['model'] = tms2_df['model'].astype(str).str.strip()

        f1_df = pd.merge(filtered_df, tms1_df, on='model')
        f2_df = pd.merge(filtered_df, tms2_df, on='model')
        final_df = pd.merge(f1_df, f2_df[['model', 'tm_w_s2', 'rm_w_s2']], on='model')

        print(final_df.head())
        logger.info(f'Alignments done. TM-align outputs saved at {self.afout_path}')
        final_outfile = self.outpath / self.jobid / f'final_df_{self.jobid}_s1-s2.csv'
        try:
            final_df.to_csv(final_outfile, index=False)
            logger.info(f'>> Identified state 1: {self.pdb_state1}')
            logger.info(f'>> Identified state 2: {self.pdb_state2}')
            logger.info(f'>> Results CSV saved at {final_outfile}')
        except Exception as e:
            logger.error(f"Failed to save results CSV: {e}")
            raise

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze generated model ensemble')
    parser.add_argument('--jobid', required=True, help='Job ID')
    parser.add_argument('--afout_path', required=True, help='Path to generated models')
    parser.add_argument('--pdb_state1', help='Reference PDB of state1')
    parser.add_argument('--pdb_state2', help='Reference PDB of state2')
    parser.add_argument('--clustering', action='store_true', help='Enable clustering of models')
    parser.add_argument('--outpath', default='results/', help='Output path')
    parser.add_argument('--ncpu', default=1, type=int, help='ncpu')

    return parser.parse_args()

def main():
    args = parse_arguments()

    analyzer = EnsembleAnalyzer(
        jobid=args.jobid,
        afout_path=args.afout_path,
        clustering=args.clustering,
        outpath=args.outpath,
        pdb_state1=args.pdb_state1,
        pdb_state2=args.pdb_state2,
        ncpu=args.ncpu
    )

    try:
        analyzer.analyze_models()
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
