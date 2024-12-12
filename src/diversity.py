#!/bin/python
# Copyright 2024 Yogesh Kalakoti (WallnerLab), Linkoping University, Sweden
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

import numpy as np
import math
import argparse
from analyse_models import EnsembleAnalyzer
import pandas as pd
import sys
import subprocess

class Projection:
    def __init__(self, base_tmscore):
        self.base_tmscore = base_tmscore
        self.line_point1 = np.array([1, base_tmscore])
        self.line_point2 = np.array([base_tmscore, 1])
        self.line_vector = self.line_point2 - self.line_point1
        self.line_vector_norm_sq = np.dot(self.line_vector, self.line_vector)

    def project(self, points):
        projs = []
        for point in points:
            point = np.array(point)
            point_vector = point - self.line_point1
            scalar_projection = np.dot(point_vector, self.line_vector) / self.line_vector_norm_sq
            vector_projection = scalar_projection * self.line_vector
            projected_point = self.line_point1 + vector_projection
            projs.append(projected_point.tolist())
        return np.array(projs)

class Quantification:
    @staticmethod
    def quantify_fill(points, range_, num_bins=100):
        min_x = points[:, 0].min()
        max_x = min_x + range_
        histogram, _ = np.histogram(points[:, 0], bins=num_bins, range=(min_x, max_x))
        non_empty_bins = np.count_nonzero(histogram)
        total_bins = num_bins
        fill_ratio = non_empty_bins / total_bins
        print(non_empty_bins, total_bins)
        return fill_ratio
    
def plotgnu(data):
    gnuplot = subprocess.Popen(['gnuplot'], stdin=subprocess.PIPE, text=True)
    commands = """
    set term dumb
    plot '-' using 1:2 with points title 'Scatter Plot'
    """
    # Write commands to gnuplot
    gnuplot.stdin.write(commands)
    for row in data:
        gnuplot.stdin.write(f"{row[0]} {row[1]}\n")
    gnuplot.stdin.write("e\n")
    gnuplot.stdin.close()
    gnuplot.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse generated model ensemble')
    parser.add_argument('--jobid', required=True, help='JOBID')
    parser.add_argument('--afout_path', required=True, help='Path to generated models')
    parser.add_argument('--pdb_state1', required=True, help='Reference PDB of state1')
    parser.add_argument('--pdb_state2', required=True, help='Reference PDB of state2')
    parser.add_argument('--outpath', default='results/', help='Output path')

    args = parser.parse_args()

    model_analyzer = EnsembleAnalyzer(jobid=args.jobid,
                                      afout_path=args.afout_path, 
                                      pdb_state1=args.pdb_state1, 
                                      pdb_state2=args.pdb_state2, 
                                      outpath=args.outpath,
                                      ncpu=1,
                                      clustering=False)
                                      
    outfile = model_analyzer.analyze_models()

    # Initialize dataframes
    fillratio_df = pd.DataFrame(index=[args.jobid])
    fillratio2_df = pd.DataFrame(index=[args.jobid])
    besttmopen_df = pd.DataFrame(index=[args.jobid])
    besttmclose_df = pd.DataFrame(index=[args.jobid])

    # Initialize Projection
    base_tmscore = 0.6
    projection_instance = Projection(base_tmscore)

    # Read data
    afout_path = args.afout_path
    df_tm = pd.read_csv(outfile)
    print(df_tm.head())
    print(df_tm.columns)

    # Get best Tm scores
    besttmo_s1 = df_tm.sort_values(by='tm_w_s1', ascending=False).reset_index().loc[0]['tm_w_s1']
    besttmo_s2 = df_tm.sort_values(by='tm_w_s2', ascending=False).reset_index().loc[0]['tm_w_s2']

    # Get fill ratio
    print(besttmo_s1, besttmo_s2)
    data = df_tm[['tm_w_s1', 'tm_w_s2']].values
    print(data)
    plotgnu(data)

    # Transform and rotate points
    transformed_points = projection_instance.project(data)
    print(transformed_points)
    # plotgnu(transformed_points)
    
    # Apply 45-degree counterclockwise rotation
    theta = np.pi / 4
    x, y = transformed_points[:, 0], transformed_points[:, 1]
    new_x = x * np.cos(theta) - y * np.sin(theta)
    new_y = x * np.sin(theta) + y * np.cos(theta)
    rotated_projection = np.array(list(zip(new_x, new_y)))
    # plotgnu(rotated_projection)

    # Calculate fill ratio
    range_ = math.dist([1, base_tmscore], [base_tmscore, 1])
    fill_ratio = Quantification.quantify_fill(rotated_projection, range_)
    print(fill_ratio)
    #distance_extreme = math.dist([1, good_examples[uniprotid][3]], [good_examples[uniprotid][3], 1])

    #fillratio_df.loc[uniprotid, rand] = fill_ratio
    #fillratio2_df.loc[uniprotid, rand] = np.round(abs(rotated_projection[:, 0].min() - rotated_projection[:, 0].max()) / distance_extreme, 3)

    #print(fillratio2_df.head())
