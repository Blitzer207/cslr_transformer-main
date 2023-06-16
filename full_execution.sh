#!/bin/bash
set -e
echo "Creating and activating environment ...."
conda create -n cslr_envrt -q -y python=3.9 --force
source activate cslr_envrt
echo "Environment creation done"
echo "Installing requirements ...."
pip install -r requirements.txt
pip install --upgrade protobuf==3.20.0
echo "Requirements installation done"
if [ -d "temp/phoenix2014-release" ]
then
    echo "Directory temp/phoenix2014-release exists, hopefully with the dataset."
else
    echo "Directory temp/phoenix2014-release does not exist."
    echo "Downloading dataset and extracting it in a temp folder"
    mkdir "temp"
    wget -c https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014.v3.tar.gz -O - | tar -xz -C temp/
    echo "Dataset download done"
fi
echo  "============Path to Environment============"
which python
echo  "==========================================="
echo "Starting preprocessing ....."
python preprocess.py > preprocessing.txt
echo "Preprocessing done"
echo "Starting training ....."
python training.py > output_transformer.txt
echo "Training done....."