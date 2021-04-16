import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from data_generator import TFData, dataset_split_loader
from lightning_network_model import TFNet
from roc_pr import ROC_PR
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import time


def parse_args():
    '''
    read inputs data from command line
    '''
    parser = argparse.ArgumentParser(description='TFNet')
    parser.add_argument('-region', type=str, required=True, help="path to the genomic region & label")
    parser.add_argument('-LR', type=float, required=True, help="learning rate")
    parser.add_argument('-output', type=str, required=True, help="path to the output folder")
    parser.add_argument('-ref', type=str, required=False, help="path to the reference genome")
    args = parser.parse_args()
    return args



#################################################
#### model training
#################################################
def TF_train(epoch_size, lr, nTF, frag_len, train_loader, val_loader, output_folder):
    # switch to GPU
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)

    train_loss_file = output_folder + "/TFNet_loss_train.txt"
    test_loss_file  = output_folder + "/TFNet_loss_test.txt"
    roc_file   = output_folder + "/TFNet_roc_test.txt"
    pr_file    = output_folder + "/TFNet_pr_test.txt"
    model_name = output_folder + "/TFNet"
    epoch_times = output_folder + "/epoch_times.txt"
    train_times = output_folder + "/train_times.txt"
    test_times = output_folder + "/test_times.txt"

    # initialize model
    net = TFNet(n_TF=nTF, frag_length=frag_len).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_func = nn.BCELoss().to(device)
    
    # start train
    
    #epochs, trains, tests = [], [], []
    for e in range(epoch_size):
        start_epoch = time.time()
        print("Start epoch[", e, "/", epoch_size, "]")
        loss_train_list = []
        loss_test_list  = []
        model_out_path  = model_name + "_" + str(e) + ".pt"
        
        start_train = time.time()
        # train
        net.train()
        for step, (seq, label) in enumerate(train_loader):
            seq   = seq.to(device)
            label = label.to(device)

            output = net(seq)

            loss = loss_func(output, label)
            loss_train_list.append(loss.to("cpu").detach().numpy())

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 100 == 0:
                print("batch:", step, "\tloss in training set:", loss_train_list[step])
        end_train = time.time()
        # test
        start_test = time.time()
        net.eval()
        for step, (seq, label) in enumerate(val_loader):
            seq   = seq.to(device)
            label = label.to(device)

            output = net(seq)
            loss_test = loss_func(output, label)
            loss_test_list.append(loss_test.to("cpu").detach().numpy())
            if step % 100 == 0:
                print("batch:", step, "\tloss in testing set:", loss_test_list[step])

            # save label and prediction to draw roc & prc
            if step == 0:
                pred_test = output.to("cpu").detach().numpy()
                label_test = label.to("cpu").numpy()
            else:
                pred_test = np.concatenate((pred_test, output.to("cpu").detach().numpy()), axis=0)
                label_test = np.concatenate((label_test, label.to("cpu").numpy()), axis=0)
        end_epoch = time.time()
          #epochs.append(end_epoch - start_epoch)
          #trains.append(end_train - start_train)
          #tests.append(end_epoch - start_test)
          
          
        # draw roc & pr
        pred_test = pred_test.transpose()
        label_test = label_test.transpose()
        label_test = label_test.astype(int)
        val_plot = ROC_PR(pred_test, label_test)

        # roc
        
        plt, mean_au_roc = val_plot.ROC(title=str(e) + 'th epoch in testing set')
        #plt.savefig(output_folder + '/ROC_' + str(e) + '.pdf')
        plt.close()

        # pr
        plt, mean_au_pr = val_plot.PR(title=str(e) + 'th epoch in testing set')
        #plt.savefig(output_folder + '/PR_' + str(e) + '.pdf')
        plt.close()
        
        # write loss
        with open(train_loss_file, 'a') as f:
            for line in loss_train_list:
                f.write(str(line) + '\t')
            f.write('\n')

        with open(test_loss_file, 'a') as f:
            for line in loss_test_list:
                f.write(str(line) + '\t')
            f.write('\n')

        # write auc
        with open(roc_file, 'a') as f:
            f.write(str(mean_au_roc) + '\n')

        with open(pr_file, 'a') as f:
            f.write(str(mean_au_pr) + '\n')
            
        # write times
        with open(epoch_times, 'a') as f:
            f.write(str(end_epoch - start_epoch) + '\n')
        
        with open(test_times, 'a') as f:
            f.write(str(end_epoch - start_test) + '\n')
        
        with open(train_times, 'a') as f:
            f.write(str(end_train - start_train) + '\n')
            
        # save model
        torch.save(net.state_dict(), model_out_path)



##############################################################
# main function
##############################################################
def main(args):
    # read args
    # args = parse_args()
    region_path = args.region
    lr = args.LR
    output_folder = args.output
    full_ref_path = "/mnt/research/compbio/wanglab/jiaxin/proj1_3d_eqtl/data/sequence/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa"
    ref_path = full_ref_path if args.ref is None else args.ref
    batch = 100
    epoch = 200

    os.system("mkdir -p {0}".format(output_folder))

    # create data sets
    seq_data = TFData(region_path, ref_path)
    nTF = seq_data.nTF
    frag_len = seq_data.frag_len

    # split the data to training & validation sets
    train_loader, val_loader = dataset_split_loader(seq_data, batch, 0.1)

    # train the model
    # TF_train(epoch, lr, nTF, frag_len, train_loader, val_loader, output_folder)
    model = TFNet(n_TF=nTF, frag_length=frag_len)
    trainer = pl.Trainer(max_epochs=epoch, profiler="simple", 
                        plugins=DDPPlugin(find_unused_parameters=False), 
                        accelerator = "ddp", replace_sampler_ddp=False, gpus = -1, num_nodes=1)
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.realpath(__file__))
    # parser = argparse.ArgumentParser(add_help=False)
    args = parse_args()
    main(args)



