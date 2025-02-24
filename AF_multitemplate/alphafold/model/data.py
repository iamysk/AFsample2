# Copyright 2021 DeepMind Technologies Limited
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

"""Convenience functions for reading data."""

import io
import os
from alphafold.model import utils
import haiku as hk
import numpy as np
# Internal import (7716).

def rename_keys(d):
    new_d = {}
    for key, value in d.items():
        parts = key.rsplit("/", 1)  # Split at the last "/"
        new_key = f"{parts[0]}//{parts[1]}" if len(parts) > 1 else key
        new_d[new_key] = rename_keys(value) if isinstance(value, dict) else value
    return new_d

def get_model_haiku_params(model_name: str, data_dir: str) -> hk.Params:
  """Get the Haiku parameters from a model name."""

  if model_name=='cfold':
    path = os.path.join(data_dir, 'params', f'params_{model_name}.npy')
    with open(path, 'rb') as f:
      params = np.load(io.BytesIO(f.read()), allow_pickle=True)
    
    params_ = rename_keys(params)
    #print(params_.keys())
    return utils.flat_params_to_haiku_cfold(params)
  else:
    path = os.path.join(data_dir, 'params', f'params_{model_name}.npz')
    with open(path, 'rb') as f:
      params = np.load(io.BytesIO(f.read()), allow_pickle=False)
    return utils.flat_params_to_haiku(params)
  
# alphafold/alphafold_iteration/distogram_head/half_logits