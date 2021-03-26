# TFNet_Optimization

## Abstract
This project focuses on optimizing PyTorch code to predict TF binding sites using DNA sequences only. The input data is the coordinates of the DNA fragments; first we get the sequences (‘ATGC’) of the fragments and encode them as a 4xn matrix. The label of each fragment is a vector, and each dimension shows whether the fragment is bound by one TF and there are totally 128 TFs. The model is a **convolutional neural network** with 1 convolutional layer, 2 dilated convolutional layers, and 1 fully connected layer that outputs the predicted labels.


My goal with this project is speeding up the training/testing of the neural network, which is already making use of a single GPU. I plan on optimizing the code on one GPU and then eventually distribute the process over multiple GPUs. I'm making heavy use of module called **Pytorch Lightning** to accomplish these goals.

## Installation

Since this is a custom network, the best way to actually get it up and running on the HPCC is to clone this git repository and run the submission script:

The other component of this project is **pytorch-lightning**, which I am using to refactor and optimize the vanilla PyTorch code. I installed this module using **conda**:

  1) The Linux installer is located at https://www.anaconda.com/products/individual at the **Anaconda Installers** section. 
  2) You can get this on your home directory by typing `wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh`
  3) After installing the installer, run it as an executable and you should be good to go! Follow the instructions at https://docs.anaconda.com/anaconda/install/linux/
  4) After the setup is finished, installing pytorch and pytorch lightning is a breeze. 
     	-Once you're in a conda environment, simply type `pip install pytorch-lightning` and `pip install torch`

## Example Code

I have included the first 50 lines of the **region.txt** file from my local directory (this file is way too big for github) for testing purposes. Also, I have included a notebook containing my initial results (titled Initial_Results.ipynb)

## Running the Submission Script

The submission script is called **run_script.sh**; here is a sample use of the code:

`sbatch run_script.sh $TRAINING_PYTHON_FILE $OUTPUT_DIRECTORY`

Note- you will need the region.txt file in the working directory for the script to work.

## References

https://pytorch-lightning.readthedocs.io/en/latest/

https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
