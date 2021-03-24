import numpy as np
import pandas as pd
import pysam
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


def seq2code(seq):
    '''
    sequence to one hot code
    :param seq: DNA sequence
    :return: one hot code
    '''
    seq_len = len(seq)
    seq_code = np.zeros((4, seq_len), dtype='float32')

    for i in range(seq_len):
        nt = seq[i]
        if nt == 'A':
            seq_code[0, i] = 1
        elif nt == 'C':
            seq_code[1, i] = 1
        elif nt == 'G':
            seq_code[2, i] = 1
        elif nt == 'T':
            seq_code[3, i] = 1
        else:
            seq_code[:, i] = 0.25

    return seq_code


def dataset_split_loader(dataset, batch_size, val_frac, shuffle_dataset=False):
    '''
    split training, validation sets into data loader
    :param dataset: pytorch dataset
    :param batch_size: int
    :param val_frac: fraction of validation data set, float
    :param shuffle_dataset: bool
    :return: train & validation data loader
    '''
    random_seed = 10
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_frac * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, shuffle=False)

    return train_loader, val_loader


class TFData(Dataset):
    def __init__(self, region_path, ref_path, shuffle_dataset=True):
        '''
        create TF dataset
        :param region_path: path of genomic region data
        :param ref_path: path of reference genome
        '''
        super(TFData,self).__init__()
        self.ref_genome = pysam.Fastafile(ref_path)
        self.data = pd.read_csv(region_path, sep='\t')
        self.nTF = self.data.shape[1]-3
        self.frag_len = self.data.iloc[0,2] - self.data.iloc[0,1]
        if shuffle_dataset:
            self.data = self.data.sample(frac=1)

    def __getitem__(self, index):
        region = self.data.iloc[index]
        region_code, label = self.region2code(region)

        region_code = torch.Tensor(region_code)
        label = torch.Tensor(label)

        return region_code, label

    def __len__(self):
        len = self.data.shape[0]
        return len

    def region2code(self, region):
        '''
        :param region: genomic region and the TF bindings
        :return: one hot code and label
        '''
        # read
        chr_id       = region[0][3:]          # just keep the number of chr
        region_start = region[1]
        region_end   = region[2]
        label        = list(region[3:])

        # coordinate to sequence
        seq_region = self.ref_genome.fetch(chr_id, region_start, region_end).upper()

        # sequence to one hot code
        region_code = seq2code(seq_region)

        return region_code, label
