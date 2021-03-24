# TFNet_Optimization

## Abstract
This project focuses on optimizing PyTorch code to predict TF binding sites. (insert description) The model is a **convolutional neural network** with 1 convolutional layer, 2 dilated convolutional layers, and 1 fully connected layer that outputs the predicted labels.

## Installation

Since this is a custom network, the best way to actually get it up and running on the HPCC is to clone this git repository and run the submission script:

The submission script is called **run_script.sh**; here is a sample use of the code

sbatch run_script.sh $TRAINING_PYTHON_FILE $OUTPUT_DIRECTORY

Note- you will need the region.txt file in the working directory for the script to work.


The other component of this project is **pytorch-lightning**, a module that I am using to refactor and optimize the vanilla PyTorch code. I installed this module using **conda**:

  1) The Linux installer is located at https://www.anaconda.com/products/individual at the **Anaconda Installers** section. 
  2) You can get this on your home directory by typing `wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh`
  3) After installing the installer, run it as an executable and you should be good to go! Follow the instructions at https://docs.anaconda.com/anaconda/install/linux/
  4) After the setup is finished, installing pytorch lightning is a breeze. Once you're in a conda environment, simply type `pip install pytorch-lightning`.

## References

https://pytorch-lightning.readthedocs.io/en/latest/
