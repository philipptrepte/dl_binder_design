# dl_binder_design

This repo contains the scripts described in the paper [Improving de novo Protein Binder Design with Deep Learning](https://www.nature.com/articles/s41467-023-38328-5).

![forgithub](https://github.com/nrbennet/dl_binder_design/assets/56419265/4c5d6a05-d2fb-4c7b-b1e0-0c743b2114eb)

# Changes to the original dl_binder_design repository



## Modified functions

### *af2_initial_guess/predict.py*

New argument `-paefilename` which defines where pae scores will be written (defaul: out.pae)

#### class AF2_runner()

- `generate_scoredict(self, feat_holder, confidences, rmsds)`

  Now adds the `binderlen` to the score_dict

#### class StructManager() 

- `record_scores(self, tag, score_dict, pae)`

  Now includes the `pae` scores to record.

### *af2_initial_guess/af2_util.py*

- `add2paefile(tag, paefilename, pae)`

  Given a pae filename, add the pae scores to the paefile.

### *helper_scripts/addFIXEDlabels.py*

Fixes `.PDB` files after using `diffuser.partial_T` in RFdiffusion, so that both chains are not `chain A` but the second chain is now `chain B`


## New functions

### *pae_clustering.py*

Performs KMeans clustering on the PAE scores to identify the most likely interaction interface region as the cluster with the minimum mean PAE. This approach is described and benchmarked in [Trepte P. & Secker C. *et al.* (2024) AI-guided pipeline for protein–protein interaction drug discovery identifies a SARS-CoV-2 inhibitor. *Mol Syst Biol*: 1-30](https://www.embopress.org/doi/full/10.1038/s44320-024-00019-8) 

```sh
pae_clustering.py -score [path/to/af2.score] -pae [path/to/af2.pae]
```

- `process_pae(i, af2scores, pae)`

  Process the PAE (Protein-Antigen Interface Energy) for a given index from a data frame containing the AF2 scores and the PAE values and performs KMeans clustering. Returns the minimum mean PAE score from the eight clusters, as well as the respective cluster size, shape and number.

- `parallel_process_pae(af2scores, pae, num_cores)`
  
  Perform parallel processing of the process_pae function on the given af2scores and pae arrays using multiple cores.

- `repair_pae_script(filepath)`

  Repairs the PAE script by running the 'repair_pae' command that calls the 'repair_pae.sh' shell script on the specified file.

  Older versions created faulty .pae files where not all pae results where in a new line. Only repairs the file if a string of the format `'(\d+(\.\d+)?)pae:'` (e.g. '11.3pae:') is found in the file. If a repair is necessary, a backup file is created. To run the `repair_pae.sh` script, make `repair_pae.sh` executable by 
  
  ```sh
  chmod +x repair_pae.sh
  ``` 
  
  then create a shortcut in a folder that is in `$PATH` using 
  
  ```sh
  sudo ln -s /path/to/af2_initial_guess/repair_pae.sh /usr/local/bin/repair_pae
  ``` 
  
  A backup file of your `.pae` file will be created as `.pae.backup`.



### *af2_initial_guess/plot_pae.py*

- `pae_heatmap(selected_binder, pae, af2scores)`

  Generate a heatmap visualization of the PAE (Positional Amino Acid Enrichment) for a selected binder.


### *af2_initial_guess/clean_af2.py*

Sometimes, the `check.point` contains in column `description` files that cannot be found in `af2.pae` or `af2.sc`. Presumambly, AF2 initial guess is not run on all files, for example because *out of memory erros occur*, or because not all pae values are stored properly. To remove such instances from `af2_check.point`, where `description` are not found in `af2.sc` files before or after running `pae_clustering.py` run `clean_af2.py` from the terminal. Define optional argument `reference` (default: `pae_description`) to check column `description` from the original `af2.sc` file or column `pae_description` from the `af2.sc` file after KMeans clustering:

```sh
clean_af2.py -score [path/to/af2.sc] -pae [path/to/af2.pae] -checkpoint [path/to/check.point] -silent [path/to/af2.silent] -checkpoint [path/to/af2_check.point] -reference [optional]
```

- `clean_checkpoint(ids, af2checkpoint)`
  
  Remove identifiers from checkpoint file `af2checkpoint` that are not in `ids`.

- `clean_af2score(af2score, af2checkpoint, af2reference='pae_description')`

  Cleans the AF2 initial guess score file by removing duplicates found in the `af2reference` column and removes the respective identifiers from the `af2checkpoint` file.

- `clean_pae(af2pae, af2checkpoint)`
  
  Cleans the PAE file by removing duplicates and removes the respective identifiers from the `af2checkpoint` file.

- `clean_silent(af2silent, af2scores_no_duplicates, pae_no_duplicates, af2checkpoint)`
  
  Cleans the silent file by removing duplicate sequences. Also removes identifiers from the `af2checkpoint` file that are found in the score and pae files, but not in the silent file and vice versa.



# Table of Contents

- [dl\_binder\_design](#dl_binder_design)
- [Changes to the original dl\_binder\_design repository](#changes-to-the-original-dl_binder_design-repository)
  - [Modified functions](#modified-functions)
    - [*af2\_initial\_guess/predict.py*](#af2_initial_guesspredictpy)
      - [class AF2\_runner()](#class-af2_runner)
      - [class StructManager()](#class-structmanager)
    - [*af2\_initial\_guess/af2\_util.py*](#af2_initial_guessaf2_utilpy)
    - [*helper\_scripts/addFIXEDlabels.py*](#helper_scriptsaddfixedlabelspy)
  - [New functions](#new-functions)
    - [*pae\_clustering.py*](#pae_clusteringpy)
    - [*af2\_initial\_guess/plot\_pae.py*](#af2_initial_guessplot_paepy)
    - [*af2\_initial\_guess/clean\_af2.py*](#af2_initial_guessclean_af2py)
- [Table of Contents](#table-of-contents)
  - [Third Party Source Code ](#third-party-source-code-)
- [Setup ](#setup-)
  - [Conda Environment ](#conda-environment-)
  - [Install ProteinMPNN-FastRelax Environment ](#install-proteinmpnn-fastrelax-environment-)
  - [Install AlphaFold2 Environment ](#install-alphafold2-environment-)
    - [Troubleshooting AF2 GPU Compatibility ](#troubleshooting-af2-gpu-compatibility-)
  - [Clone ProteinMPNN ](#clone-proteinmpnn-)
  - [Silent Tools ](#silent-tools-)
  - [Download AlphaFold2 Model Weights ](#download-alphafold2-model-weights-)
- [Inference ](#inference-)
  - [Summary ](#summary-)
  - [Example Commands ](#example-commands-)
  - [ProteinMPNN-FastRelax Binder Design ](#proteinmpnn-fastrelax-binder-design-)
    - [Running ProteinMPNN with Fixed Residues ](#running-proteinmpnn-with-fixed-residues-)
  - [AlphaFold2 Complex Prediction ](#alphafold2-complex-prediction-)
- [Troubleshooting ](#troubleshooting-)


## Third Party Source Code <a name="sourcecode"></a>
This repository provides a copy of Brian Coventry's silent tools, these tools are also provided [here](https://github.com/bcov77/silent_tools). This repository provides a wrapper to [Justas' ProteinMPNN code](https://github.com/dauparas/ProteinMPNN) and some of the code in the wrapper class is adapted directly from ProteinMPNN code. Finally, this repository provides a version of the AlphaFold2 source code with the "initial guess" modifications described in [this paper](https://www.nature.com/articles/s41467-023-38328-5). The AF2 source code is provided with the original DeepMind license at the top of each file.

# Setup <a name="setup0"></a>

## Conda Environment <a name="setup1"></a>

I have split the single conda env that this repo used to use (dl_binder_design.yml; still provided in <base_dir>/include if anyone is interested) into two smaller and easier to install environments. The old environment required PyTorch, JAX, TensorFlow, and PyRosetta packages to all be compatible with one another, which is difficult. The new environments should be easier to install and I have also added import tests so that it is easier and faster to check that the installation has been successful.

Both of these environments require PyRosetta which requires a license that is free to academics and available [here](https://graylab.jhu.edu/pyrosetta/downloads/documentation/PyRosetta_Install_Tutorial.pdf). This license will give you access to the USERNAME and PASSWORD referenced below. If you do not provide this USERNAME and PASSWORD, you will get a CondaHTTPError when you attempt to run the installation.

The steps to installing the environments are as follows:

- Ensure that you have the Anaconda or Miniconda package manager
- Ensure that you have the PyRosetta channel included in your `~/.condarc`
- Your `~/.condarc` should look something like this:
```
channels: 
- https://USERNAME:PASSWORD@conda.graylab.jhu.edu
- conda-forge
- defaults
```
- More information about conda installing PyRosetta may be found here: https://www.pyrosetta.org/downloads
- Clone this repo

## Install ProteinMPNN-FastRelax Environment <a name="setup1.1"></a>
- Navigate to <base_dir>/include
- Run `conda env create -f proteinmpnn_fastrelax.yml`
- Test the environment by activating your environment and running `python importtests/proteinmpnn_importtest.py`, if you encounter an error in this script then something has gone wrong with your installation. If the script prints out that the tests pass then you have installed this environment correctly. This script will also test whether it can access a GPU, it is not recommended to run ProteinMPNN with a GPU as it is only marginally faster than running on CPU and FastRelax cannot take advantage of the GPU anyway.

## Install AlphaFold2 Environment <a name="setup1.2"></a>
- Navigate to <base_dir>/include
- Run `conda env create -f af2_binder_design.yml`
- Test the environment by activating your environment and running `python importtests/af2_importtest.py`. Run this script from a node which has acccess to the GPU you wish to use for AF2 inference, this script will print out a message about whether it was able to find and use the GPU on your node. If this script hits an error before printing anything then the installation has not been done correctly.

### Troubleshooting AF2 GPU Compatibility <a name="setup1.3"></a>

Getting a conda environment that recognizes and can run AF2 on your GPU is one of the more difficult parts of this process. Because of the many different GPUs out there, it is not possible for us to provide one .yml file that will work with all GPUs. We provide a CUDA 11.1 compatible env (dl_binder_design.yml) and a CUDA 12 compatible env (af2_binder_design.yml). For other versions of the CUDA driver, you may need to change which CUDA version is installed in the conda env, this can be done by changing the CUDA version in this line of af2_binder_design.yml: `- jax[cuda12_pip]`.

NOTE: This conda environment can only accomodate NVIDIA GPUs at this point in time.

## Clone ProteinMPNN <a name="setup2"></a>
This repo requires the code from ProteinMPNN to work. It expects this code to be in the mpnn_fr directory so we can just clone it to be there
- Naviate to <base_dir>/mpnn_fr
- Run `git clone https://github.com/dauparas/ProteinMPNN.git`

## Silent Tools <a name="setup3"></a>
The scripts contained in this repository work with a type of file called silent files. These are essentially a bunch of compressed .pdb files that are all stored in one file. Working with silent files is conventient and saves a lot of disk space when dealing with many thousands of structures.

Brian Coventry wrote a bunch of really nice commandline tools (called silent_tools) to manipulate silent files. These tools are included in this repository but may also be downloaded separately from [this](https://github.com/bcov77/silent_tools) GitHub repo.

The two commands that allow you to go from pdb to silent file and back are the following:

pdbs to silent: `<base_dir>/silentfrompdbs *.pdb > my_designs.silent` 

silent to pdbs: `<base_dir>/silentextract all_structs.silent`

NOTE: Some silent tools require PyRosetta and will fail if run in a Python environment without access to PyRosetta.

## Download AlphaFold2 Model Weights <a name="setup4"></a>
The scripts in this repository expect AF2 weights to be in <base_dir>/model_weights/params and will fail to run if the weights are not there. If you already have AF2 params_model_1_ptm.npz weights downloaded then you may simply copy them to <base_dir>/model_weights/params or create a symlink. If you do not have these weights downloaded you will have to download them from DeepMind's repository, this can be done as follows:

```
cd <base_dir>/af2_initial_guess
mkdir -p model_weights/params && cd model_weights/params
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar --extract --verbose --file=alphafold_params_2022-12-06.tar 
```

# Inference <a name="inf0"></a>

## Summary <a name="inf0.5"></a>

The binder design pipeline requires protein binder backbones as an input. The recommended way to generate these backbones is to use [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion), which will give you a directory of .pdb files. These files can either be turned into silent files which are more memory efficient and easier to work with than a directory of pdb files (and your system administrator will thank you for using them), or you can use these directories of .pdb files as-is with this pipeline, using either the `-pdbdir` flag alone or the `-pdbdir` flag in combination with the `-runlist` flag.

## Example Commands <a name="inf0.5"></a>

Example commands demonstrating how to run each of these scripts with different types of input can be found here:

```
<base_dir>/examples
```

## ProteinMPNN-FastRelax Binder Design <a name="inf1"></a>

Here is an example of how to run ProteinMPNN-FastRelax with a silent file of designs. This will use the default of 1 FastRelax cycle of 1 ProteinMPNN sequence per round (NOTE: running with -relax_cycles > 0 and -seqs_per_struct > 1 is disabled as it leads to an explosion in the amount of FastRelax trajectories being run and is probably bad idea):

```
<base_dir>/mpnn_fr/dl_interface_design.py -silent my_designs.silent
```

This will create a file titled `out.silent` containing your designs. This file can be fed directly to AF2 interface prediction.

With the refactor, this script is now able to read and write both PDB files and silent files. The script I have also added more informative argument messages which can be accessed by running:

```
<base_dir>/mpnn_fr/dl_interface_design.py -h
```

NOTE: This script expects your binder design to be the first chain it receives. This script is robust to non-unique indices, unlike the AF2 interface script.
NOTE 2: The outputs of this script do not have relaxed sidechains (since sidechains are not input to AF2 and it's not worth the computation to relax them) so the structures will look strange if you visualize them in PyMol, this is perfectly normal, the structures will look better after run though AF2.

### Running ProteinMPNN with Fixed Residues <a name="inf2"></a>

If you used RFdiffusion to generate your binder designs and would like to fix a region, you can use the following command to add 'FIXED' labels to your pdbs which will be recognized by the ProteinMPNN scripts. Thanks to Preetham Venkatesh for writing this!

```
python <base_dir>/helper_scripts/addFIXEDlabels.py --pdbdir /dir/of/pdbs --trbdir /dir/of/trbs --verbose
```

These pdb files can be collected into a silent file (or just used as PDB files) and run through the ProteinMPNN script which will detect the FIXED labels and keep those sequence positions fixed.


## AlphaFold2 Complex Prediction <a name="inf3"></a>

Running the interface prediction script is simple:

```
<base_dir>/af2_initial_guess/predict.py -silent my_designs.silent
```

This will create a file titled `out.silent` containing the AF2 predictions of your designs. It will also output a file titled `out.sc` with the scores of the designs, `pae_interaction` is the score that showed the most predictivity in the experiments performed in the paper.

With the refactor, this script is now able to read and write PDB and silent files as well as perform both monomer and complex predictions. The arguments for these can be listed by running:

```
<base_dir>/af2_initial_guess/predict.py -h
```

NOTE: This script expects your binder design to be the first chain it receives. The binder will be predicted from single sequence and with an intial guess. The target chains will be fixed to the input structure. The script also expects your residue indices to be unique, ie. your binder and target cannot both start with residue 1.

# Troubleshooting <a name="trb0"></a>

One of the most common errors that people have been having is one that looks like this:

```
Struct with tag SAMETAG failed in 0 seconds with error: <class 'EXCEPTION'>
```

Where SAMETAG and EXCEPTION can be many different things. What is happening here is that the main loops of both of the scripts provided here are wrapped in a try-catch block; the script tries to run each design and if an error occurs, the script notes which design had an error and continues to the next design. This error catching is convenient when running production-scale design campaigns but is a nuisance for debugging since the messages are not very informative.

If you hit this error, I recommend running the same command that yielded the error but while adding the `-debug` flag to the command. This flag will make the script run without the try-catch block and errors will print with the standard verbose, easier-to-debug messages.



