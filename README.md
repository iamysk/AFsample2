# AFsample2
Introducing a way to induce diversity in the AF2 ensemble by spanning the conformational ensemble and identifying possible states.
![20240226_mov.gif](20240226_mov.gif)

## Introduction

AFsample2 is a generative protein structure prediction system based on AF2 that is able to induce significant conformational diversity for a given protein.

See article preprint:
[AFsample2: Predicting multiple conformations and ensembles with AlphaFold2](https://www.biorxiv.org/content/10.1101/2024.05.28.596195v1)


## Installation

1. [Install Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/)
2. Setup environment

```
# Clone this repository
git clone https://github.com/iamysk/AFsample2.git
cd AFsample2/

# install dependencies
conda env create -n <env_name> --file=environment.yaml
conda activate <env_name>
python -m pip install -r requirements.txt
```
3. Make sure that all sequence databases are available at ```<data_path>```. Follow the official AlphaFold guide [here](https://github.com/google-deepmind/alphafold) to set up databases. 
```bash
cd scripts
chmod +x download_all_data.sh
./download_all_data.sh <data_path> reduced_dbs
```

4. [OPTIONAL] Install Rosetta suite for clustering tasks from here [Download page](https://rosettacommons.org/software/download/). Make sure that a C++ compiler is installed. 

```bash
## Optional. Ignore if compilers already installed
$ sudo apt-get install build-essential      # install C++ compilers

## Unzip tarball and compile
tar -xvzf rosetta[releasenumber].tar.gz
cd rosetta*/main/source
./scons.py -j <num_cores> mode=release bin/rosetta_scripts.mpi.linuxgccrelease       # Significiantly fast with multithreading
```
Refer to this [guide](https://new.rosettacommons.org/demos/latest/tutorials/install_build/install_build#installing-rosetta) for further details.

## Usage

Step-by-step instructions to (1) generate model ensembles (2) Analyze diversity and (3) Clustering and downstream analysis

### Ensemble generation
Follow the steps to generate a diverse conformational ensemble for a given ```<fasta_path>```. 
```bash
'''
Inputs: 
<method>: Method to run among ['afsample2', 'speachaf', 'af2', 'msasubsampling']
<fasta_paths>: path to .fasta file
<flagfile> : AF2 specific parameter file
<nstruct>: Number of structures to generate
<msa_rand_fraction>: % MSA randomization in random msa_perturbation_mode
<models_to_use>: (Optional) AF2 model to use (model_1, model_2 ...)

# Outputs:
# <output_dir>: Path to output directory
'''

# Example usage (AFsample2)
python AF_multitemplate/run_afsample2.py --method afsample2 \
		--fasta_paths examples/P31133/P31133.fasta \
		--flagfile AF_multitemplate/monomer_full_dbs.flag \
		--nstruct 1 \
		--msa_rand_fraction 0.20 \
		--model_preset=monomer \
		--output_dir examples/	

```
Other useful flags (run ```<AF_multitemplate/run_alphafold.py --help>``` for more details)
| flag | Options | Usage |
| --- | --- | --- |
| --msa_perturbation_mode| random, profile | To choose MSA perturbation mode |
| --use_precomputed_features| Bool| Whether to use precomputed features.pkl file |




###  Container usage usage
```bash
# Docker
docker pull kyogesh/afsample2:mark8

# Docker usage
docker run --volume <path-to-databases>:/databases \
           --volume <path-to-inputs>:/inputs \
           --volume <path-to-outputs>:/outputs \
           -it kyogesh/afsample2:mark8 \
           --method afsample2     \
           --fasta_paths inputs/example.fasta     \
           --flagfile inputs/flagfile.flag     \
           --nstruct 4     \
           --msa_rand_fraction 0.20     \
           --model_preset=monomer     \
           --output_dir examples/
```

```bash
# Apptainer
apptainer pull docker://kyogesh/afsample2:mark8

apptainer run --nv \
    -B <database_path>:/databases \
    -B examples/:/input \
    -B AF_multitemplate:/app/alphafold/AF_multitemplate \
    afsample2_mark8.sif \
    --method afsample2 \
    --fasta_paths /input/P31133/P31133.fasta \
    --flagfile /app/alphafold/AF_multitemplate/monomer_dbs_full_dbs_container.flag \
    --nstruct 10 \
    --model_preset monomer \
    --output_dir /input/ \
    --use_precomputed_features=True \
    --dropout=True
```

## Diversity analysis and state identification
```bash
'''
Inputs: 
<afout_path>: Path to generated models
<pdb_state1>: Reference PDB of state1
<pdb_state1>: Reference PDB of state1
<ncpu>: number of cores to use

# Outputs:
# final_df_ref1-ref2.csv file saved at results/
'''

# Example usage (If references available)
python src/analyse_models.py --method afsample2 \
	--protein 8E6Y \
	--afout_path examples/8E6Y/ \
	--pdb_state1 examples/8E6Y/referencea/2fs1_A.pdb \
	--pdb_state2 examples/8E6Y/referencea/8e6y_A.pdb \
	--clustering=False	\
	--ncpu=16
```

OUTPUT:
```bash
     ___    ______                           __    ___ 
    /   |  / ____/________ _____ ___  ____  / /__ |__ 
   / /| | / /_  / ___/ __ `/ __ `__ \/ __ \/ / _ \__/ /
  / ___ |/ __/ (__  ) /_/ / / / / / / /_/ / /  __/ __/ 
 /_/  |_/_/   /____/\__,_/_/ /_/ /_/ .___/_/\___/____/ 
                                  /_/                  
     
2025-01-08 14:23:02,328 [INFO] Analyzing models...
2025-01-08 14:23:02,328 [INFO] Reference state1, state2: examples/8E6Y/referencea/2fs1_A.pdb, examples/8E6Y/referencea/8e6y_A.pdb
2025-01-08 14:23:02,329 [INFO] Found 10 models in examples/8E6Y
2025-01-08 14:23:02,708 [INFO] Low confidence (mean plddt<50) residue indices: []
2025-01-08 14:23:02,711 [INFO] Most confident model: examples/8E6Y/unrelaxed_model_1_pred_4_dropout.pdb, Confidence: 86.42021052631578
examples/8E6Y/referencea/2fs1_A.pdb examples/8E6Y/referencea/8e6y_A.pdb
2025-01-08 14:23:02,712 [INFO] Received reference PDBs: examples/8E6Y/referencea/2fs1_A.pdb, examples/8E6Y/referencea/8e6y_A.pdb
TM-align (examples/8E6Y/referencea/2fs1_A.pdb - models): 100%|██| 10/10 [00:00<00:00, 245280.94it/s]
TM-align (examples/8E6Y/referencea/8e6y_A.pdb - models): 100%|██| 10/10 [00:00<00:00, 170500.16it/s]
2025-01-08 14:23:03,116 [INFO] Alignments done. TM-align outputs saved at examples/8E6Y
2025-01-08 14:23:03,126 [INFO] >> State 1: examples/8E6Y/referencea/2fs1_A.pdb
2025-01-08 14:23:03,126 [INFO] >> State 2: examples/8E6Y/referencea/8e6y_A.pdb
2025-01-08 14:23:03,126 [INFO] >> Results CSV saved at results/afsample2/final_df_8E6Y_s1-s2.csv
```

```
# Example usage (If references NOT available)
python src/analyse_models.py --method afsample2 --protein 8E6Y --afout_path examples/8E6Y/  --clustering=False--ncpu=16   
```

## Datasets and reproducibility

All data and scripts required to generate the plots in the manuscript are provided [here](https://zenodo.org/records/14534088?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY0MGY1OThlLWIxN2QtNDNmNy05NGE0LTg1NWFiYzE3MTIyNSIsImRhdGEiOnt9LCJyYW5kb20iOiI4NmY5MDU5NjQ3MWQyZmFlZWE1ZWFlYmIwYjRjMGE0ZSJ9.G0VvOWpri5ehL2BephKUC0F0FoSnVuxCUalJtIPeQVkVCAnVnip6Ob89uJchmwYMpKizpAB-v30zVCTie59rww). An overview of the directory structure, along with a description of each folder and its contents is provided in the dataset page. Extract as follows.

```
tar --use-compress-program=unzstd -xvf input_datasets.tar.zst

└── input_datasets
    ├── oc23
    │   ├── fastas					
    │   ├── filtered_dict.pickle	# pdbids and stats for states
    │   ├── msas					# in .pkl format
    │   └── pdbs	
    └── tp16
        ├── fastas
        ├── filtered_dict.pickle
        ├── msas
        └── pdbs

tar --use-compress-program=unzstd -xvf generated_models.tar.zst

└── generated_models
   ├── oc23
   │   ├── afsample2
   │   ├── SPEACH_AF
   │   ├── ...
   └── tp16
       ├── afsample2
       ├── SPEACH_AF
       ├── ...

tar --use-compress-program=unzstd -xvf analysis_results.tar.zst

└── analysis_results
   ├── oc23
   │   ├── afsample2
   │   ├── SPEACH_AF
   │   ├── ...
   └── tp16
       ├── afsample2
       ├── SPEACH_AF
       ├── ...
```

## How to Cite
```
@article {Kalakoti2024.05.28.596195,
	author = {Kalakoti, Yogesh and Wallner, Bj{\"o}rn},
	title = {AFsample2: Predicting multiple conformations and ensembles with AlphaFold2},
	elocation-id = {2024.05.28.596195},
	year = {2024},
	doi = {10.1101/2024.05.28.596195},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/06/02/2024.05.28.596195},
	eprint = {https://www.biorxiv.org/content/early/2024/06/02/2024.05.28.596195.full.pdf},
	journal = {bioRxiv}
}

@article{Wallner2023,
	title = {AFsample: improving multimer prediction with AlphaFold using massive sampling},
	volume = {39},
	ISSN = {1367-4811},
	url = {http://dx.doi.org/10.1093/bioinformatics/btad573},
	DOI = {10.1093/bioinformatics/btad573},
	number = {9},
	journal = {Bioinformatics},
	publisher = {Oxford University Press (OUP)},
	author = {Wallner,  Bj\"{o}rn},
	editor = {Kelso,  Janet},
	year = {2023},
	month = sep 
}
```

## License

APACHE
