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
# Inputs: 
# <models_to_use>: Path to generated models
# <fasta_paths>: Reference PDB of state1
# <msa_rand_fraction>: Reference PDB of state1

# Outputs:
# <output_dir>: Path to output directory
'''

# Example usage
python AF_multitemplate/run_alphafold.py --models_to_use model_1
	--fasta_paths example/8E6Y/8E6Y.fasta      
	--output_dir example/8E6Y
	--msa_rand_fraction 0.1
	--flagfile AFmultitemplate/monomer_full_dbs.flag

```

### Diversity analysis and state identification

Analyse model ensemble to generate diversity plot if reference is available. In case references are not available, identify possible states.

```bash
'''
# Inputs: 
# <afout_path>: Path to generated models
# <pdb_state1>: Reference PDB of state1
# <pdb_state1>: Reference PDB of state1
# <ncpu>: number of cores to use

# Outputs:
# final_df_ref1-ref2.csv file saved at results/
'''

# Example usage (If references available)
python src/analyse_models.py --afout_path examples/8E6Y/ \
	--pdb_state1 examples/8E6Y/referencea/2fs1_A.pdb \
	--pdb_state2 examples/8E6Y/referencea/8e6y_A.pdb \
	--jobid 8E6Y \
	--clustering=False
	--ncpu=16

# Example usage (If references not available)
python src/analyse_models.py --jobid 8E6Y --afout_path examples/8E6Y/ --clustering=False --ncpu=16      

```

### Clustering and reference-free state determination
```bash
$ pip install af_sample2

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
