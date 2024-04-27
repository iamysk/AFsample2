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

import numpy as np

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
        return fill_ratio

if __name__ == "__main__":
    # Load data
    with open('../master_data/filtered_dict.pickle', 'rb') as handle:
        filtered_dict = pickle.load(handle)

    # Filter good examples
    good_examples = {k: v for k, v in filtered_dict.items() if v is not None and v[3] < 0.85}

    # Rand values
    rands = ['000', '00', '15']

    # Initialize dataframes
    fillratio_df = pd.DataFrame(index=list(good_examples.keys()), columns=rands)
    fillratio2_df = pd.DataFrame(index=list(good_examples.keys()), columns=rands)
    besttmopen_df = pd.DataFrame(index=list(good_examples.keys()), columns=rands)
    besttmclose_df = pd.DataFrame(index=list(good_examples.keys()), columns=rands)

    # Initialize Projection
    projection_instance = Projection(base_tmscore)

    # Loop through good examples
    for uniprotid in good_examples:
        for rand in rands:
            # Read data
            afout_path = f'data/af_io_abl_{rand}/'
            df_tm = pd.read_csv(f'{afout_path}{uniprotid}/final_df.csv')

            # Get best Tm scores
            besttmopen = df_tm.sort_values(by='TM_open', ascending=False).reset_index().loc[0]['TM_open']
            besttmclose = df_tm.sort_values(by='TM_close', ascending=False).reset_index().loc[0]['TM_close']
            besttmopen_df.loc[uniprotid, rand] = besttmopen
            besttmclose_df.loc[uniprotid, rand] = besttmclose

            # Get fill ratio
            data = df_tm[['TM_open', 'TM_close']].values
            filtered_data = data[(data[:, 0] >= good_examples[uniprotid][3]) & (data[:, 1] >= good_examples[uniprotid][3])]

            if len(filtered_data) == 0:
                continue

            transformed_points = projection_instance.project(filtered_data)

            # Apply 45-degree counterclockwise rotation
            theta = np.pi / 4
            x, y = transformed_points[:, 0], transformed_points[:, 1]
            new_x = x * np.cos(theta) - y * np.sin(theta)
            new_y = x * np.sin(theta) + y * np.cos(theta)
            rotated_projection = np.array(list(zip(new_x, new_y)))

            # Calculate fill ratio
            range_ = math.dist([1, good_examples[uniprotid][3]], [good_examples[uniprotid][3], 1])
            fill_ratio = Quantification.quantify_fill(rotated_projection, range_)
            distance_extreme = math.dist([1, good_examples[uniprotid][3]], [good_examples[uniprotid][3], 1])

            fillratio_df.loc[uniprotid, rand] = fill_ratio
            fillratio2_df.loc[uniprotid, rand] = np.round(abs(rotated_projection[:, 0].min() - rotated_projection[:, 0].max()) / distance_extreme, 3)
