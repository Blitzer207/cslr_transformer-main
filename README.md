# Transformer based CSLR system for German sign Language 

This repository contains code to build a German CSLR system based on transformers and the [RWTH-PHOENIX-Weather 2014: Continuous Sign Language Recognition Dataset](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/).

## Hardware Requirements
> **Warning:**
>  The transformer model has around 110 millions trainable parameters. Therefore, be cautious with hardware.

1. At least 150Gb Disk space
2. At least 24Gb GPU Memory if you intend to use GPU.


## Software Requirements

1. Make sure you have [Anaconada](https://www.anaconda.com/products/distribution) installed.
2. Open a conda command line (you should land in the base environment).


## Full Execution
To perform the experiments in the order "environment creation > preprocessing > feature extraction > processing and evaluation" run the following command:
``bash full_execution.sh``

If you are interested in the step by step execution, follow the steps under the upcoming sections.

## Setup
Recommended Linux OS with  Python version 3.9 
1. Download the dataset from [here](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) and extract it.


2. Create a new conda-environment and activate it.
   * ``conda create -n slr_env python=3.9``
   * ``conda activate slr_env``


3. Install all requirements. (Please manually install cuda accordingly if you plan to use a GPU)
   * ``pip install -r  requirements.txt``
   * ``pip install --upgrade protobuf==3.20.0``


4. To visualize the hyperparameter optimization results, install [DeepCAVE ](https://automl.github.io/DeepCAVE/main/installation.html). The tool was tested only on Linux and Mac.

> **Note:**
>  The dataset has 2 subsets. One for multi signer and the other for signer independent. Therefore, all simulation scripts contain a "multisigner" boolean argument to specify which subset to use.


## Preprocessing and feature extraction
The preprocessing script does the following:

1. Generates a numpy array containing informations on each dataset samples.
2. Generates the vocabulary of the dataset. A word_distribution.txt file is generated.
3. Performs feature extraction  and label encoding from all data splits and save features and labels in numpy arrays.


The script requires following arguments: path_to_dataset ( string indicating the path to the "phoenix2014-release" folder), multisigner (boolean to indicate if we use multisigner or signer independent). It can be launched via : ``python preprocess.py --path_to_dataset path --no-multisigner``


## Training

Assuming features were extracted and saved, the training script can be started by specifying following arguments:
- epochs: integer
- multisigner: Boolean
- batch_size: integer
- seed: Integer
- device: String (cuda or cpu)
- optimizer: String Adam or SGD
- evaluation_during_training: Boolean
- evaluation_frequency: Integer
- save_per_epoch: Boolean

Other options can directly be changed in the scripts. Be cautious while attempting this.

Results are plotted and stored.

For Example, One could use the following command:

``python training.py --epochs 10 --batch_size 50 --no-multisigner > output_transformer.txt``


## Testing

To test the trained model, just call the testing script with following arguments:
- path_to_model: The path to the model you want to test.
- multisigner: Boolean
- batch-size: integer
- device: String (cuda or cpu)


Results are generated and saved in a file
An Example of command could be:
``python testing.py --no-multisigner --batch_size 4 --path_to_model transformer_model.pt --device cpu``


> **Note:**
> Fill free to contact me in case you have some issues or questions.
> 