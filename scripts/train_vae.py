from json import encoder
import os
import time
from xmlrpc.client import boolean
import torch
import pickle
import argparse
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

from src.nri_encoder import MLPEncoder
from src.nri_decoder import MLPDecoder
from src.utils import mask, load_data, gumbel_softmax, my_softmax, kl_categorical_uniform


def parse_args():
    """Parse argumetns

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser(description="code for training VAE")
    parser.add_argument('-n',
                        '--nodes',
                        dest='nodes',
                        type=int,
                        help="Number of nodes in the interacting system",
                        default=5)
    parser.add_argument(
        '-sd',
        '--sep_hidden_dims',
        dest='sep_hidden_dims',
        type=int,
        help=
        "Dimension of hidden states in the edge-type specific Neural network",
        default=256)
    parser.add_argument(
        '-so',
        '--sep_out_dims',
        dest='sep_out_dims',
        type=int,
        help="Dimension of out in the edge-type specific Neural network",
        default=256)
    parser.add_argument('-d',
                        '--dims',
                        dest='node_dims',
                        type=int,
                        help="Dims of each node",
                        default=4)
    parser.add_argument('-pred_s',
                        '--pred_step',
                        dest='pred_steps',
                        type=int,
                        help="Prediction time steps",
                        default=1)
    parser.add_argument('-e',
                        '--epoch',
                        dest='epoch_num',
                        type=int,
                        help="Number of trainning epochs",
                        default=40)
    parser.add_argument('-hid',
                        '--hidden',
                        dest='hidden_dims',
                        type=int,
                        help="Dimension of hidden layer in the encoder MLP",
                        default=256)
    parser.add_argument('-ts', '--time_steps', type=int,dest='time_steps', default=49)
    parser.add_argument('-et',
                        '--edge_types',
                        dest='edge_types',
                        type=int,
                        help="Number of edge types",
                        default=2)

    # added action to --cuda argument, this will give a bool value rather than a string or
    # integer, and changed the default to false. With action='store_true', you can use it as a flag,
    # just use --cuda to enable cuda
    parser.add_argument('-cuda',
                        '--cuda',
                        dest='cuda',
                        default=False,
                        help='Use this flag if you want to run on GPU',
                        action='store_true')
    parser.add_argument('-dr',
                        '--dropout_rate',
                        dest='dropout',
                        type=float,
                        help="Dropout rate",
                        default=0.01)
    return parser.parse_args()


def reconstruction_error(pred, target):
    """This function computes the error between prediction trajectory and target trajectory.

    Args:
        pred:
        target:

    Returns:
        Mean prediction error.
    """
    loss = ((pred - target)**2)
    return loss.sum() / (loss.size(0) * loss.size(1))


def train(args):
    """Train and validate the encoder model

    Args:
        args: pased argument. See 'def parse_args()' for details

    """

    # Designate number of training epochs, optimizer, scheduler and criterion (loss function)
    epoch_nums = args.epoch_num
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    # timestr will be used in model path.
    timestr = time.strftime("%Y%m%d_%H%M")
    parent_folder_enc = f'../saved_model/encoder/{timestr}'
    parent_folder_dec = f'../saved_model/decoder/{timestr}'

    # Create a directory if not exist.
    if not os.path.exists(parent_folder_enc):
        os.mkdir(parent_folder_enc)
    # Create a directory if not exist.
    if not os.path.exists(parent_folder_dec):
        os.mkdir(parent_folder_dec)

    best_acc = -1
    for i in range(epoch_nums):
        loss_kl_train = []
        loss_rec_train = []
        acc_train = []
        loss_kl_val = []
        loss_rec_val = []
        acc_val = []
        #  encoder.train() and decoder.train() tell your model that you are training the model
        encoder.train()
        decoder.train()
        # The path for trained model, with timestr and epoch index i.
        # If model is improved, the model will be saved to this direcotry.
        model_path = f'/{timestr}/{i}/'

        for batch_index, (input_batch, target) in enumerate(train_loader):
            # forward pass
            # print(input_batch.shape)

            input_batch = input_batch.to(device)
            target = target.to(device)

            # Shape of `logits`: [batch_size, #nodes * (#nodes - 1), #edge_types]
            logits = encoder(input_batch, send_mask, rec_mask)
            
            # Sampling
            edges = gumbel_softmax(logits)
            prob = my_softmax(logits, -1)
            
            output = decoder(input_batch, edges, send_mask, rec_mask, args.pred_steps)

            target_traj = input_batch[:, :, 1:, :]
            
            loss_reconstrcution = reconstruction_error(output, target_traj)
            
            # While the KL term for a uniform prior is just the sum of entropies (plus a constant)
            # https://arxiv.org/abs/1802.04687
            loss_kl = kl_categorical_uniform(prob, args.nodes, args.edge_types)

            loss = loss_reconstrcution + loss_kl
            
            # Set the gradients to zero before starting to do backpropragation
            optimizer.zero_grad()
            loss.backward()

            # All optimizers implement a step() method, that updates the parameters.
            # Learning rate scheduling should be applied after optimizer’s update.
            optimizer.step()
            scheduler.step()

            # The predicted edge type is the one with largest value
            pred = logits.data.max(2, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            acc = correct / (target.size(0) * target.size(1))

            # Append results
            loss_kl_train.append(loss_kl.item())
            loss_rec_train.append(loss_reconstrcution.item())
            acc_train.append(acc.item())

    # Now tell the model that I want to test it
        encoder.eval()
        decoder.eval()
        for batch_index, (input_batch, target) in enumerate(valid_loader):

            input_batch = input_batch.to(device)
            target = target.to(device)

            logits = encoder(input_batch, send_mask, rec_mask)
            
            # Sampling
            edges = gumbel_softmax(logits)
            prob = my_softmax(logits, -1)
            
            output = decoder(input_batch, edges, send_mask, rec_mask, args.pred_steps)

            target_traj = input_batch[:, :, 1:, :]
            
            loss_reconstrcution = reconstruction_error(output, target_traj)
            
            # While the KL term for a uniform prior is just the sum of entropies (plus a constant)
            # https://arxiv.org/abs/1802.04687
            loss_kl = kl_categorical_uniform(prob, args.nodes, args.edge_types)

            loss = loss_reconstrcution + loss_kl
            
            pred = logits.data.max(2, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            acc = correct / (target.size(0) * target.size(1))

            loss_rec_val.append(loss_reconstrcution.item())
            loss_kl_val.append(loss_kl.item())
            acc_val.append(acc.item())

        # Save the model if the model performance is improved.
        # Print out messages
        if np.mean(acc_val) > best_acc:
            best_acc = np.mean(acc_val)
            best_model_path_enc = "../saved_model/encoder" + model_path
            best_model_path_dec = "../saved_model/decoder" + model_path

            if not os.path.exists(best_model_path_enc):
                os.mkdir(best_model_path_enc)
            if not os.path.exists(best_model_path_dec):
                os.mkdir(best_model_path_dec)

            torch.save(encoder.state_dict(), best_model_path_enc + 'model.ckpt')
            torch.save(decoder.state_dict(), best_model_path_dec + 'model.ckpt')

            print('-----------------------------------------------')
            print(
                f'epoch {i} VAE training finishes. Model performance improved.'
            )
            print(f'Validation acc {np.mean(acc_val)}')
            print(f'Validation recover loss {np.mean(loss_rec_val)}')

            print(f'save best model to {best_model_path_enc}, {best_model_path_dec}')

    return best_model_path_enc, best_model_path_dec


#----------------------------------------------
def test(args, best_model_path_enc, best_model_path_dec):
    loss_kl_test = []
    loss_rec_test = []
    acc_test = []
    encoder.eval()
    decoder.eval()
    # Put parameters into the network
    encoder.load_state_dict(torch.load(best_model_path_enc + 'model.ckpt'))
    decoder.load_state_dict(torch.load(best_model_path_dec + 'model.ckpt'))

    for batch_idx, (data, target) in enumerate(test_loader):
        data = data[:, :, :args.time_steps, :]  # .contiguous()
        input_batch = data.to(device)
        target = target.to(device)

        logits = encoder(input_batch, send_mask, rec_mask)
            
        # Sampling
        edges = gumbel_softmax(logits)
        prob = my_softmax(logits, -1)
            
        output = decoder(input_batch, edges, send_mask, rec_mask, args.pred_steps)

        target_traj = input_batch[:, :, 1:, :]
            
        loss_reconstrcution = reconstruction_error(output, target_traj)
            
        # While the KL term for a uniform prior is just the sum of entropies (plus a constant)
        # https://arxiv.org/abs/1802.04687
        loss_kl = kl_categorical_uniform(prob, args.nodes, args.edge_types)

        loss = loss_reconstrcution + loss_kl
            
        pred = output.data.max(2, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        acc = correct / (target.size(0) * target.size(1))

        loss_kl_test.append(loss_kl.item())
        loss_rec_test.append(loss_reconstrcution.item())
        acc_test.append(acc.item())    

    print('-------------testing finish-----------------')
    print(f'load model from: {best_model_path_enc},{best_model_path_dec}')
    print(f'test kl loss: {np.mean(loss_kl_test)}')
    print(f'test reconstruction loss: {np.mean(loss_rec_test)}')
    print(f'test acc: {np.mean(acc_test)}')


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)  # can be safely used even for CPU runs, ignored silently

    # args, model, loaders are Global variable
    args = parse_args()

    # get the device based on availability and args flag
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    send_mask, rec_mask = mask(args.nodes)
    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
        batch_size=10, suffix='_springsLight5', root=False)
    encoder = MLPEncoder(args.time_steps * args.node_dims, args.hidden_dims,
                       args.edge_types, args.dropout).to(device)
    
    decoder = MLPDecoder(args.node_dims, args.sep_hidden_dims, args.sep_out_dims,
                        args.edge_types, args.hidden_dims, args.dropout)
    
    send_mask = send_mask.to(device)
    rec_mask = rec_mask.to(device)

    if device == 'cuda':
        print('Run in GPU.')
    else:
        print('No GPU provided.')

    best_model_path_enc, best_model_path_dec = train(args)
    test(args, best_model_path_enc, best_model_path_dec)
