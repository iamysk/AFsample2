# AFsample2
![20240226_mov.gif](20240226_mov.gif)

## Introduction

AFsample2 is a generative protein strutcre prediction system based on AF2 that is able to induce significant conformational diversity for a given protein.

## Installation

1. [Install Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/)
2. Setup environment

```
# Clone this repository
git clone https://github.com/iamysk/AFsample2.git
cd AFunmasked/

# install dependencies
conda env create -n <env_name> --file=environment.yaml
conda activate <env_name>
python -m pip install -r requirements.txt
```
3. Make sure that all sequence databases are available. Follow the official AlphaFold guide [here](https://docs.anaconda.com/free/miniconda/miniconda-install/).
```bash
$ cd scripts
$ chmod +x download_all_data.sh
$ ./download_all_data.sh <data_path> reduced_dbs
```
4. Install Rosetta suite for clustering tasks ([Download link](https://en.wikipedia.org/wiki/Tar_(computing))). Make sure that a C++ compiler is installed. 

```bash
## Optional. Ignore if compielrs already installed
$ sudo apt-get install build-essential      # To install C++ compilers

## Unzip tarball and compile
$ tar -xvzf rosetta[releasenumber].tar.gz
$ cd rosetta*/main/source
$ ./scons.py -j <num_cores> mode=release bin/rosetta_scripts.mpi.linuxgccrelease       # Significiantly fast with multithreading

Refer to this [guide](https://new.rosettacommons.org/demos/latest/tutorials/install_build/install_build#installing-rosetta) for further details.
```

## Usage

Step-by-step instructions to (1) generate model ensembles (2) Analyze diversity and (3) Clustering and downstream analysis

### Ensemble generation
Follow the steps to generate a diverse conformational ensemble for a given ```<fasta_path>```. 
```bash
$ pip install af_sample2
$ sh AF_multitemplate/abl_msa.sh --models_to_use <models_to_use>        # default=model_1 
                                 --fasta_paths <fasta_path>         
                                 --output_dir <output_dir> 
                                 --msa_rand_fraction <Random masking>   # default=0.1
                                 --flagfile <flag_file>                 # default = AFmultitemplate/monomer_full_dbs.flag

```
Explain how to use AFsample2. Provide examples or code snippets to demonstrate its usage. Include any configuration options or parameters that can be customized by the user.

### Diversity analysis
```bash
$ pip install af_sample2
$ sh AF_multitemplate/abl_msa.sh --models_to_use <models_to_use>        # default=model_1 
                                 --fasta_paths <fasta_path>         
                                 --output_dir <output_dir> 
                                 --msa_rand_fraction <Random masking>   # default=0.1
                                 --flagfile <flag_file>                 # default = AFmultitemplate/monomer_full_dbs.flag

```

### Clustering and reference-free state determiantion
```bash
$ pip install af_sample2
$ sh AF_multitemplate/abl_msa.sh --models_to_use <models_to_use>        # default=model_1 
                                 --fasta_paths <fasta_path>         
                                 --output_dir <output_dir> 
                                 --msa_rand_fraction <Random masking>   # default=0.1
                                 --flagfile <flag_file>                 # default = AFmultitemplate/monomer_full_dbs.flag

```

## How to Cite

If AFsample2 is research-related or if you'd like users to cite it in their work, provide a citation format here. Include the title, authors, publication year (if applicable), and any relevant details such as conference or journal name.

## Contributing

Provide guidelines for contributing to AFsample2. This may include instructions for reporting bugs, suggesting new features, or submitting pull requests. Be sure to mention any coding standards or conventions to follow.

## License

Specify the license under which AFsample2 is distributed. Include any terms and conditions associated with the license.

## Acknowledgements

Acknowledge any individuals, organizations, or projects that have contributed to AFsample2 or inspired its development.

## Contact

Provide contact information for the maintainers of AFsample2 in case users have questions, feedback, or need support.

## Additional Resources

Include links to any additional resources related to AFsample2, such as documentation, tutorials, or demo videos.
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |


## Project Status

Optional: Provide information about the current status of AFsample2, such as whether it's actively maintained, in development, or no longer maintained.

