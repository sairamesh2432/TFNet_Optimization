import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim



class TFNet(pl.LightningModule):
    def __init__(self, n_TF, frag_length, conv_kernel_size=8, conv_pooling=4, dilation_level=2,
                 dilation_pooling=4, dropout_level=0.3, bias_label=True):
        '''
        :param n_TF:
        :param frag_length:
        :param conv_kernel_size:
        :param conv_pooling:
        :param dilation_level:
        :param dilation_pooling:
        :param dropout_level:
        :param bias_label:
        '''
        super(TFNet, self).__init__()
        self.n_TF = n_TF
        self.frag_length = frag_length
        self.conv_kernel_size = conv_kernel_size
        self.conv_pooling = conv_pooling
        self.dilation_level = dilation_level
        self.dilation_pooling = dilation_pooling
        self.dropout_level = dropout_level
        self.bias_label = bias_label
        self.linear_in = self.outLength()

        # 1 layer Convolution
        self.Conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=n_TF, kernel_size=conv_kernel_size, bias=bias_label),
            nn.BatchNorm1d(num_features=n_TF),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=conv_pooling),
            nn.Dropout(p=dropout_level),
        )

        # 2 layer dilation Convolution
        self.Dilat1 = nn.Sequential(
            nn.Conv1d(in_channels=n_TF, out_channels=n_TF, kernel_size=conv_kernel_size,
                      dilation=dilation_level, bias=bias_label),
            nn.BatchNorm1d(num_features=n_TF),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=dilation_pooling),
            nn.Dropout(p=dropout_level),
        )

        self.Dilat2 = nn.Sequential(
            nn.Conv1d(in_channels=n_TF, out_channels=n_TF, kernel_size=conv_kernel_size,
                      dilation=dilation_level, bias=bias_label),
            nn.BatchNorm1d(num_features=n_TF),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=dilation_pooling),
            nn.Dropout(p=dropout_level),
        )

        # 2 layer fc
        self.Linear = nn.Sequential(
            nn.Linear(in_features=n_TF * self.linear_in, out_features=200, bias=bias_label),
            nn.Dropout(p=dropout_level),
            nn.LeakyReLU(),

            nn.Linear(in_features=200, out_features=n_TF, bias=bias_label),
            nn.Sigmoid(),

        )


    def forward(self, seq):
        '''
        :param seq: one hot code
        :return: prediction of TF bindings
        '''
        x = self.Conv(seq)
        x = self.Dilat1(x)
        x = self.Dilat2(x)
        # flat
        x = x.view(x.size(0), -1)
        out = self.Linear(x)

        return out


    def outLength(self):
        '''
        CNN: L_out = L_in - dilation * (kernel_size - 1)
        pooling: L_out = L_in - (kernel_size - 1) / kernel_size
        :return: the dimension of fc inputs
        '''
        out_length = self.frag_length - self.conv_kernel_size + 1
        out_length = int(np.ceil((out_length - self.conv_pooling + 1) / self.conv_pooling))

        for i in range(2):
            out_length = out_length - self.dilation_level * (self.conv_kernel_size - 1)
            out_length = int(np.ceil((out_length - self.dilation_pooling + 1) / self.dilation_pooling))

        return out_length
    
    @staticmethod
    def BCELoss(output, label):
        return nn.BCELoss(output, label)
    
    def training_step(self, batch, batch_idx):
        seq, label = batch
        output = self.forward(seq)
        loss = self.BCELoss(output, label)
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        seq, label = val_batch
        output = self.forward(seq)
        loss = self.BCELoss(output, label)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': output, 'label':label}
        
    def configure_optimizers(self, lr=0.001):
      optimizer = optim.Adam(self.parameters(), lr=lr)
      return optimizer

