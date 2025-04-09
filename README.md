# Generative prediction of causal gene sets responsible for complex traits

This repository contains two Jupyter notebooks and a Python script needed to reproduce the results of "Generative prediction of causal gene sets responsible
for complex traits" to be published.

## Installing necessary software 
The software packages needed to run the code may be installed by following these steps:

1. Navigate to the top-level directory of this repository.
2. Execute the command  ```conda env create -f environment.yml```. This should install the necessary software dependencies in a virtual environment.
3. Execute the command ```conda activate my_env```. This initiates the virtual environment that was just installed.

Now you should be able to run the code!

## Download the necessary data
There are several data files that are too large to host on GitHub. They are currently available at on Google Drive [https://drive.google.com/drive/folders/1_H66cbaQ5b0b8PE_XHILDVhLP8XNjaj3?usp=sharing].

The data consist of:
1. single-cell RNAseq data on the human complex disease traits featured in the manuscript (labeled by GEO series, see Table 1 in main text)
2. transcriptional responses to gene perturbations
3. gene_dict and matching_indices needed to run the jupyter notebooks below
4. sample optimization data for allergic asthma trait to run the second jupyter notebook TWAVE2 below: data_aa_lam_3.zip

Note: these data take up a couple GB of space, so downloads could take a long time for slow internet connections.

## Run the code

The code consists of the following files:
1. TWAVE1_training_eigengenes_average_optimization.ipynb --- a Jupyter notebook that trains the Variational Autoencoder TWAVE on the source data, reduces dimension to the causal eigengenes, and performs the optimization for the average states.
2. TWAVE2_optimization_analysis.ipynb --- a Juptyer notebook that assembles gene sets from the point-to-point optimization into gene perturbation co-occurrence networks using the maximum entropy model as a null model.
3. TWAVE_optimization.py --- a Python script that parallelizes the point-to-point constrained optimization. We recommend running this on a cluster since each of the 2500 x 2 optimizations takes between 15 and 30 minutes on a single processor.

## Need help?
Please reach out to Ben Kuznets-Speck at biophysben@gmail.com with any questions.
