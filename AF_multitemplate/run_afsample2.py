# Copyright 2021 DeepMind Technologies Limited
# Copyright 2024 WallnerLab (Linköping University, Sweden)
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

"""Full AlphaFold protein structure prediction script."""
import enum
import json
import os
import pathlib
from pathlib import Path
import pickle
import random
import shutil
import sys
import time
from typing import Any, Dict, Mapping, Union
import copy

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config, config_cfold
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax
import jax.numpy as jnp
import numpy as np

# Internal import (7716).

logging.set_verbosity(logging.INFO)

@enum.unique
class ModelsToRelax(enum.Enum):
  ALL = 0
  BEST = 1
  NONE = 2

flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')

flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('jackhmmer_binary_path', shutil.which('jackhmmer'),
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', shutil.which('hhblits'),
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', shutil.which('hhsearch'),
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', shutil.which('hmmsearch'),
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', shutil.which('hmmbuild'),
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('kalign_binary_path', shutil.which('kalign'),
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniref30_database_path', None, 'Path to the UniRef30 '
                    'database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', None, 'Path to the Uniprot '
                    'database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'Path to the PDB '
                    'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('db_preset', 'full_dbs',
                  ['full_dbs', 'reduced_dbs'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs)')
flags.DEFINE_enum('model_preset', 'monomer', list(config.MODEL_PRESETS.keys())+list(config_cfold.MODEL_PRESETS.keys()),
                  #['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer', 'multimer_v1', 'multimer_v2', 'multimer_v3'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('num_multimer_predictions_per_model', 5, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. '
                     'Note: this FLAG only applies if model_preset=multimer')
flags.DEFINE_integer('num_monomer_predictions_per_model', 1, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per monomer model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. '
                     'Note: this FLAG only applies if model_preset=monomer')
flags.DEFINE_integer('nstruct',1,'How many predictions to generate')
flags.DEFINE_integer('nstruct_start', 1, 'model to start with, can be used to parallelize jobs, '
                     'e.g --nstruct 20 --nstruct_start 20 will only make model _20'
                     'e.g --nstruct 21 --nstruct_start 20 will make model _20 and _21 etc.')
flags.DEFINE_boolean('use_precomputed_msas', True, 'Whether to read MSAs that '
                     'have been written to disk instead of running the MSA '
                     'tools. The MSA files are looked up in the output '
                     'directory, so it must stay the same between multiple '
                     'runs that are to reuse the MSAs. WARNING: This will not '
                     'check if the sequence, database or configuration have '
                     'changed.')
flags.DEFINE_boolean('seq_only', False,'Exit after sequence searches')
flags.DEFINE_boolean('no_templates', False,'Will skip templates faster than using the date')
flags.DEFINE_integer('max_recycles', 3,'Max recycles')
flags.DEFINE_integer('uniprot_max_hits', 50000, 'Max hits in uniprot MSA')
flags.DEFINE_integer('mgnify_max_hits', 500, 'Max hits in uniprot MSA')
flags.DEFINE_integer('uniref_max_hits', 10000, 'Max hits in uniprot MSA')
flags.DEFINE_integer('bfd_max_hits', 10000, 'Max hits in uniprot MSA')
flags.DEFINE_float('early_stop_tolerance', 0.5,'early stopping threshold')
flags.DEFINE_enum_class('models_to_relax', ModelsToRelax.BEST, ModelsToRelax,
                        'The models to run the final relaxation step on. '
                        'If `all`, all models are relaxed, which may be time '
                        'consuming. If `best`, only the most confident model '
                        'is relaxed. If `none`, relaxation is not run. Turning '
                        'off relaxation might result in predictions with '
                        'distracting stereochemical violations but might help '
                        'in case you are having issues with the relaxation '
                        'stage.')
flags.DEFINE_boolean('use_gpu_relax', None, 'Whether to relax on GPU. '
                     'Relax on GPU can be much faster than CPU, so it is '
                     'recommended to enable if possible. GPUs must be available'
                     ' if this setting is enabled.')
flags.DEFINE_boolean('dropout', False, 'Turn on drop out during inference to get more diversity')
flags.DEFINE_boolean('cross_chain_templates', False, 'Whether to include cross-chain distances in multimer templates')
flags.DEFINE_boolean('cross_chain_templates_only', False, 'Whether to include cross-chain distances in multimer templates')
flags.DEFINE_boolean('separate_homomer_msas', False, 'Whether to force separate processing of homomer MSAs')
flags.DEFINE_list('models_to_use', None, 'specify which models in model_preset that should be run')
flags.DEFINE_float('msa_rand_fraction', 0, 'Level of MSA randomization (0-1)', lower_bound=0, upper_bound=1)
flags.DEFINE_enum('method', 'afsample2', ['afsample2', 'speachaf', 'af2', 'msasubsampling', 'cfold'], 'Choose method from <afsample2, speachaf, af2, cfold>')
flags.DEFINE_enum('msa_perturbation_mode', 'random', ['random', 'profile'], 'msa_perturbation_mode')
flags.DEFINE_string('msa_perturbation_profile', None, 'A file containing the frequency for the residues that could be randomized')
flags.DEFINE_boolean('use_precomputed_features', False, 'Whether to use precomputed msafeatures')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')


def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
  """Recursively changes jax arrays to numpy arrays."""
  for k, v in output.items():
    if isinstance(v, dict):
      output[k] = _jnp_to_np(v)
    elif isinstance(v, jnp.ndarray):
      output[k] = np.array(v)
  return output

def read_rand_profile():
  msa_frac={}
  logging.info(f'Reading msa_perturbation_profile from {FLAGS.msa_perturbation_profile}')
  with open(FLAGS.msa_perturbation_profile,'r') as f:
    for line in f.readlines():
      (pos,frac)=line.rstrip().split()
      msa_frac[int(pos)]=float(frac)
  return msa_frac

def get_columns_to_randomize(msa, method):
  nres = msa.shape[1]
  if method=='afsample2':
    if FLAGS.msa_perturbation_mode=='random':
      if FLAGS.msa_rand_fraction:
        columns_to_randomize = np.random.choice(range(0, nres), size=int(nres*FLAGS.msa_rand_fraction), replace=False) # Without replacement
      else:
        logging.info(f'Error! --msa_rand_fraction required for "{FLAGS.msa_perturbation_mode}" mode. Exiting...')
        sys.exit()

    elif FLAGS.msa_perturbation_mode=='profile':
      logging.info(f'Perturbing MSA with custom profile')
      columns_to_randomize=[]
      if FLAGS.msa_perturbation_profile!=None:
        msa_frac = read_rand_profile()
        for pos in msa_frac:
          r = np.random.random()
          if msa_frac[pos]>r:
            columns_to_randomize.append(pos-1)
      else:
        logging.info(f'Error! --msa_perturbation_profile required for "profile" mode. Exiting...')
        sys.exit()
  
  if method=='speachaf':
    logging.info(f'Perturbing MSA with "speachaf profile"')
    columns_to_randomize=[]
    if FLAGS.msa_perturbation_profile!=None:
      msa_frac = read_rand_profile()
      for pos in msa_frac:
        r = np.random.random()
        if msa_frac[pos]>r:
          columns_to_randomize.append(pos-1)
    else:
      logging.info(f'Error! --msa_perturbation_profile required for speachaf. Exiting...')
      sys.exit()
  return columns_to_randomize

def display_perturbations(residue_indices, protein_length, chunk_size=50):
  protein_representation = ['-' for _ in range(protein_length)]
  for pos in residue_indices:
      protein_representation[pos] = '*'
  
  logging.info("Protein sequence (Perturbed positions marked with '*'):")
  for i in range(0, protein_length, chunk_size):
      chunk = protein_representation[i:i + chunk_size]
      logging.info(f"{i:3}-{i+chunk_size-1:3}: {''.join(chunk)}")
  return None


def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int,
    models_to_relax: ModelsToRelax):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Get features.
  t_0 = time.time()
  if not FLAGS.use_precomputed_features:
    feature_dict = data_pipeline.process(input_fasta_path=fasta_path, msa_output_dir=msa_output_dir)

    timings['features'] = time.time() - t_0
    
    # Write out features as a pickled dictionary
    features_output_path = os.path.join(output_dir, 'features.pkl')
    with open(features_output_path, 'wb') as f:
      pickle.dump(feature_dict, f, protocol=4)

    # Kill if seq_only==True (after saving feature.pkl)
    if FLAGS.seq_only:
      logging.info('Exiting since --seq_only is True... ')
      sys.exit()
  
  else:
    logging.info('Using precomputed msa features...')
    with open(f'{output_dir}/features.pkl', 'rb') as handle:
      feature_dict = pickle.load(handle)
    
    # Check feat integrity (for CFOLD)
    keycheck = ['aatype', 'between_segment_residues', 'domain_name', 'residue_index', 'seq_length', 'sequence', 'deletion_matrix_int', 'msa', 'num_alignments', 'msa_species_identifiers', 'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions', 'template_domain_names', 'template_sequence', 'template_sum_probs']
    # Add empty keys to run properly (Fix for cfold feature files)
    for key in keycheck:
      if 'template' in key:
        feature_dict[key] = []

  num_models = len(model_runners)
  logging.info(model_runners.keys())
  #print(feature_dict.keys())
  
  for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
    # unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    # if os.path.exists(unrelaxed_pdb_path):
    #   # print(f'Model exists. {unrelaxed_pdb_path}')
    #   logging.info(f'Model exists. {unrelaxed_pdb_path}')
    #   continue

    # initialize run
    logging.info('Initializing model %s on %s', model_name, fasta_name)
    t_0 = time.time()
      
    #model_random_seed = model_index + random_seed * num_models
    rand_fd = copy.deepcopy(feature_dict)
    msa = rand_fd['msa']

    ###################################
    # AFSAMPLE2
    ###################################
    # Reference for aa codes -> IDS (https://github.com/iamysk/AFsample2/blob/38fba468f5e5031e1b65481cf8fe74ffc04b2b64/AF_multitemplate/alphafold/common/residue_constants.py#L633)
    if FLAGS.method=='afsample2':
      model_random_seed = model_index + random_seed * num_models
      logging.info(f'Running AFsample2, Substitution: X (Unknown), Randomization {FLAGS.msa_rand_fraction} %')
      # Check is model file exists
      Path(f"{output_dir}/{FLAGS.method}").mkdir(parents=True, exist_ok=True)
      unrelaxed_pdb_path = os.path.join(output_dir, FLAGS.method, f'unrelaxed_{model_name}.pdb')
      if os.path.exists(unrelaxed_pdb_path): logging.info(f'Model exists: {unrelaxed_pdb_path}'); continue

      columns_to_randomize = get_columns_to_randomize(msa, FLAGS.method)
      display_perturbations(columns_to_randomize, msa.shape[1])
      for col in columns_to_randomize:
        msa[1:, col] = np.array([20]*(rand_fd['msa'].shape[0]-1))  # Replace MSA columns with X (20)
      rand_fd['msa'] = msa

      processed_feature_dict = model_runner.process_features(rand_fd, random_seed=model_random_seed)
      t_0 = time.time()
      prediction_result = model_runner.predict(processed_feature_dict, random_seed=model_random_seed)
      timings[f'process_features_{model_name}'] = time.time() - t_0
      logging.info(
        'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
        model_name, fasta_name, timings[f'process_features_{model_name}'])
      
      save_results(output_dir, model_name, prediction_result, processed_feature_dict, unrelaxed_pdb_path, model_runner, columns_to_randomize)

    ###################################
    # SPEACHAF
    ###################################
    elif FLAGS.method=='speachaf':
      logging.info(f'Running SPEACH_AF, Substitution: A (Alanine)')
      model_random_seed = model_index + random_seed * num_models
      # Reading all available perturbation profiles
      if os.path.isdir(FLAGS.msa_perturbation_profile):
        profiles = glob.glob(FLAGS.msa_perturbation_profile+f'/{fasta_name}_*.txt')
        logging.info(f'Found {len(profiles)} perturbation profiles')
        for i, profile in enumerate(profiles):
          # Check if model is already predicted
          Path(f"{output_dir}/sp{i}").mkdir(parents=True, exist_ok=True)
          unrelaxed_pdb_path = os.path.join(f"{output_dir}/sp{i}", f'unrelaxed_{model_name}.pdb')
          if os.path.exists(unrelaxed_pdb_path): logging.info(f'Model exists: {unrelaxed_pdb_path}'); continue

          model_random_seed = model_index + random_seed * num_models
          columns_to_randomize = get_columns_to_randomize(msa, profile)
          logging.info(f'Using {profile}')
          logging.info(f'Perturbing positions {columns_to_randomize}')
          for col in columns_to_randomize:
            realcol = msa[:, col]
            realcol[realcol != 21] = 0
            msa[:, col] = realcol  # Replace MSA columns, including first sequence, excluding gaps (-) with A (0)

          rand_fd['msa']=msa
          processed_feature_dict = model_runner.process_features(rand_fd, random_seed=model_random_seed)
          t_0 = time.time()
          prediction_result = model_runner.predict(processed_feature_dict, random_seed=model_random_seed)
          # make 
          save_results(f"{output_dir}/sp{i}", model_name, prediction_result, processed_feature_dict, unrelaxed_pdb_path, model_runner, columns_to_randomize)
      else:
        logging.info(f'ERROR with provided profiles...')

    ################################### 
    # AFvanilla
    ###################################
    elif FLAGS.method=='af2':   # No randomization
      model_random_seed = model_index + random_seed * num_models
      logging.info(f'mNo MSA perturbation. Running at default values.\n')
      processed_feature_dict = model_runner.process_features(feature_dict, random_seed=model_random_seed)
      columns_to_randomize=None
      t_0 = time.time()
      prediction_result = model_runner.predict(processed_feature_dict, random_seed=model_random_seed)
      
      # Log timings
      timings[f'process_features_{model_name}'] = time.time() - t_0
      logging.info(
        'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
        model_name, fasta_name, timings[f'process_features_{model_name}'])

      unrelaxed_pdb_path = os.path.join(output_dir, FLAGS.method, f'unrelaxed_{model_name}.pdb')
      # Check is model file exists
      if os.path.exists(unrelaxed_pdb_path): logging.info(f'Model exists: {unrelaxed_pdb_path}'); continue
      save_results(output_dir, model_name, prediction_result, processed_feature_dict, unrelaxed_pdb_path, model_runner, columns_to_randomize)
    
    ###################################
    # MSAsubsampling
    ###################################
    elif FLAGS.method=='msasubsampling':
      logging.info(f'Running {FLAGS.method}')
      # Check is model file exists
      Path(f"{output_dir}/{FLAGS.method}").mkdir(parents=True, exist_ok=True)
      unrelaxed_pdb_path = os.path.join(output_dir, FLAGS.method, f'unrelaxed_{model_name}.pdb')
      if os.path.exists(unrelaxed_pdb_path): logging.info(f'Model exists: {unrelaxed_pdb_path}'); continue
      model_random_seed = model_index + random_seed * num_models
      processed_feature_dict = model_runner.process_features(feature_dict, random_seed=model_random_seed)
      columns_to_randomize=None
      t_0 = time.time()
      prediction_result = model_runner.predict(processed_feature_dict, random_seed=model_random_seed)

      # Log timings
      timings[f'process_features_{model_name}'] = time.time() - t_0
      logging.info(
        'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
        model_name, fasta_name, timings[f'process_features_{model_name}'])

      columns_to_randomize=None
      save_results(output_dir, model_name, prediction_result, processed_feature_dict, unrelaxed_pdb_path, model_runner, columns_to_randomize)

    else:
      logging.info(f'Incorrect method, select from ["afsample2", "speachaf", "af2"]')
      logging.info('Exiting!!')
      sys.exit(1)

    if benchmark:
      t_0 = time.time()
      model_runner.predict(processed_feature_dict,
                           random_seed=model_random_seed)
      t_diff = time.time() - t_0
      timings[f'predict_benchmark_{model_name}'] = t_diff
      logging.info(
          'Total JAX model %s on %s predict time (excludes compilation time): %.1fs',
          model_name, fasta_name, t_diff)

def save_results(output_dir, model_name, prediction_result, processed_feature_dict, unrelaxed_pdb_path, model_runner, columns_to_randomize):
    plddt = prediction_result['plddt']

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, FLAGS.method, f'result_{model_name}.pkl')
    # Remove jax dependency from results.
    np_prediction_result = _jnp_to_np(dict(prediction_result))

    with open(result_output_path, 'wb') as f:
      keys_to_remove=['distogram', 'experimentally_resolved', 'masked_msa','aligned_confidence_probs']
      # keys_to_remove=['experimentally_resolved', 'masked_msa','aligned_confidence_probs']
      for k in keys_to_remove:
        if k in np_prediction_result:
          del(np_prediction_result[k])
      
      if FLAGS.method=='afsample2':
        np_prediction_result['X_msa_indexes']=columns_to_randomize
      # np_prediction_result['perturbed_msa']=rand_fd['msa']
      pickle.dump(np_prediction_result, f, protocol=4)

    # Save json fle with metrics
    json_out=result_output_path+'.json'
    d_keep={}
    keys=['ranking_confidence','ptm','iptm']
    for key in keys:
        if key in np_prediction_result.keys():
            d_keep[key]=float(np_prediction_result[key])

    with open(json_out, "w") as f:
        json.dump(d_keep, f)
    
    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode)

    unrelaxed_pdb = protein.to_pdb(unrelaxed_protein)
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(unrelaxed_pdb)
    f.close()

    return None

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for tool_name in (
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
    if not FLAGS[f'{tool_name}_binary_path'].value:
      raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                       'sure it is installed on your system.')

  use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
  _check_flag('small_bfd_database_path', 'db_preset',
              should_be_set=use_small_bfd)
  _check_flag('bfd_database_path', 'db_preset',
              should_be_set=not use_small_bfd)
  _check_flag('uniref30_database_path', 'db_preset',
              should_be_set=not use_small_bfd)

  run_multimer_system = 'multimer' in FLAGS.model_preset
  _check_flag('pdb70_database_path', 'model_preset',
              should_be_set=not run_multimer_system)
  _check_flag('pdb_seqres_database_path', 'model_preset',
              should_be_set=run_multimer_system)
  _check_flag('uniprot_database_path', 'model_preset',
              should_be_set=run_multimer_system)

  if FLAGS.model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  if run_multimer_system:
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        database_path=FLAGS.pdb_seqres_database_path)
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
  else:
    template_searcher = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        databases=[FLAGS.pdb70_database_path])
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  monomer_data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniref30_database_path=FLAGS.uniref30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd,
      use_precomputed_msas=FLAGS.use_precomputed_msas,
      mgnify_max_hits=FLAGS.mgnify_max_hits,
      uniref_max_hits=FLAGS.uniref_max_hits,
      bfd_max_hits=FLAGS.bfd_max_hits,
      no_templates=FLAGS.no_templates)

  if run_multimer_system:
    num_predictions_per_model = max(FLAGS.nstruct,FLAGS.num_multimer_predictions_per_model)
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        uniprot_database_path=FLAGS.uniprot_database_path,
        use_precomputed_msas=FLAGS.use_precomputed_msas,
        max_uniprot_hits=FLAGS.uniprot_max_hits,
        separate_homomer_msas=FLAGS.separate_homomer_msas)
  else:
    num_predictions_per_model = max(FLAGS.nstruct,FLAGS.num_monomer_predictions_per_model)
    data_pipeline = monomer_data_pipeline

  model_runners = {}
  if FLAGS.model_preset=='cfold_monomer':
    model_names = config_cfold.MODEL_PRESETS[FLAGS.model_preset]
    print(model_names)
  else:
    model_names = config.MODEL_PRESETS[FLAGS.model_preset]
  if FLAGS.models_to_use:
    model_names =[m for m in model_names if m in FLAGS.models_to_use]
  if len(model_names)==0:
    raise ValueError(f'No models to run: {FLAGS.models_to_use} is not in {config.MODEL_PRESETS[FLAGS.model_preset]}')
  for model_name in model_names:
    if FLAGS.model_preset=='cfold_monomer':
      model_config = config_cfold.model_config(model_name)
    else:
      model_config = config.model_config(model_name)

    if run_multimer_system:
      model_config.model.num_ensemble_eval = num_ensemble
      if FLAGS.cross_chain_templates:
        logging.info("Turning cross-chain templates ON (use at your own risk)")
        model_config.model.embeddings_and_evoformer.cross_chain_templates = True
      if FLAGS.cross_chain_templates_only:
        logging.info("Turning cross-chain templates ON, in-chain templates OFF (use at your own risk)")
        model_config.model.embeddings_and_evoformer.cross_chain_templates = False
        model_config.model.embeddings_and_evoformer.cross_chain_templates_only = True
    else:
      model_config.data.eval.num_ensemble = num_ensemble
      model_config.data.common.num_recycle = FLAGS.max_recycles #bw IMPORTANT needed for monomer pipeline 20240518
    model_config.model.num_recycle = FLAGS.max_recycles
    model_config.model.global_config.eval_dropout = FLAGS.dropout
    model_config.model.recycle_early_stop_tolerance=FLAGS.early_stop_tolerance

    logging.info(f'Setting max_recycles to {model_config.model.num_recycle}')
    logging.info(f'Setting early stop tolerance to {model_config.model.recycle_early_stop_tolerance}')
    logging.info(f'Setting dropout to {model_config.model.global_config.eval_dropout}')
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    
    model_runner = model.RunModel(model_config, model_params, cfold=True)

    if FLAGS.method=='msasubsampling':
      # MSA subsampling implementaion (https://elifesciences.org/articles/75751
      for max_extra_msa in [16, 32, 64, 128, 256, 512, 1024, 5120]:
        for i in range(FLAGS.nstruct_start, int((num_predictions_per_model+1)/8)):
          model_runner.config.data.common.max_extra_msa = int(max_extra_msa)
          model_runner.config.data.eval.max_msa_clusters = int(min(max_extra_msa/2, 512))
          model_runners[f'{model_name}_pred_{i}_{max_extra_msa}'] = model_runner
    else:
      for i in range(FLAGS.nstruct_start, num_predictions_per_model+1):
        model_runners[f'{model_name}_pred_{i}'] = model_runner
  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  amber_relaxer = relax.AmberRelaxation(
      max_iterations=RELAX_MAX_ITERATIONS,
      tolerance=RELAX_ENERGY_TOLERANCE,
      stiffness=RELAX_STIFFNESS,
      exclude_residues=RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
      use_gpu=FLAGS.use_gpu_relax)

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_runners))
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure for each of the sequences.
  for i, fasta_path in enumerate(FLAGS.fasta_paths):
    fasta_name = fasta_names[i]
    predict_structure(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed,
        models_to_relax=FLAGS.models_to_relax)

if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'data_dir',
      'uniref90_database_path',
      'mgnify_database_path',
      'template_mmcif_dir',
      'max_template_date',
      'obsolete_pdbs_path',
      'use_gpu_relax',
  ])

  app.run(main)
