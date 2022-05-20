# d-reflex
D-reflex experiments

## python
python scripts and dataset 

### dataset
situations dataset of 2000 samples used in the revised version of the paper (with damage combinations and reject everything but the hand contact) 

### scripts
written using JupyterLab and Jupytext.

#### t22_04_20(NN_training).py
Jupytext script to train and evaluate a neural network using the collected dataset.

#### 22_05_05(reaction_delay).py
Jupytext script to evaluate the effect of the delay before activating the reflex 

#### 22_05_06(analyse_dataset).ipynb
jupyter notebook to analyse the origin of the unavoidable/avoidable using a decision tree

#### 22_05_07(generate_data).py
Jupytext script to colect data using the c++ WBC and Dart simulation. 

#### 22_05_09(prepare_data_for_NN).ipynb
jupyter script to clean the dataset for the NN training 

#### 22_05_16(RAL_Revision_for_paper_video).py
Jupytext script to generate the simulation videos for the paper 

#### 22_05_16(RAL_revision_friction).py
Jupytext script to evaluate the effect of the friction

#### 22_05_16(RAL_revision_main_figure).py
Jupytext script  to generate the snapshots for the main figure 

## cpp

### talos 
urdf files used for the experiments.

### d-reflex 
custom tasks, behaviors and controller for D-Reflex.

### inria_wbc
Whole-Body Controller for the TALOS. 
