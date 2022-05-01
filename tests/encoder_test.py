# This script tests if input data can flow through the trained encoder model successfully 
# Modify the argument based on your trained model.

import time
import torch

import numpy as np
from torch import nn
import argparse
from src.utils import mask, load_data
from src.nri_encoder import MLPEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, dest='model_path', default='../saved_model/encoder/20220422_0942/9/',
                    help='Specified model path within saved_model/encoder folder')
parser.add_argument('-ds', '--data_suffix', dest='data_suffix', default='_springsLight5')
parser.add_argument('-n', '--nodes', dest='nodes', default=5)
parser.add_argument('-d', '--node_dims', dest='nodes_dims', default=4)
parser.add_argument('-hid', '--hidden', dest='hidden_dims', default=256)
parser.add_argument('-ts', '--time_steps', dest='time_steps', default=49)  # 49->4
parser.add_argument('-et', '--edge_types', dest='edge_types', default=2)
parser.add_argument('-cuda', '--cuda', dest='cuda', default=False, action='store_true')
parser.add_argument('-dr', '--dropout_rate', dest='dropout', default=0.01)
parser.add_argument('-b', '--batch_size', dest='batch_size', default=5)

# Set random seed
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

args = parser.parse_args()

# get the device based on availability and args flag
device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

send_mask, rec_mask = mask(args.nodes)

model = MLPEncoder(args.time_steps * args.nodes_dims, args.hidden_dims, args.edge_types, args.dropout)
model.load_state_dict(torch.load(args.model_path + 'model.ckpt', map_location='cpu'))

model = model.to(device)

send_mask = send_mask.to(device)
rec_mask = rec_mask.to(device)

if device == 'cuda':
    print('Run in GPU')
else:
    print('No GPU provided.')

# Comment the following codes if your are using Dataloader class
test_series = np.load("../data/test.npy")
test_edges = np.load("../data/edge_type.npy")
test_series = torch.tensor(test_series).to(device)
test_edges = torch.tensor(test_edges).to(device)
print('Data loader generated')

data = test_series[:, :, :args.time_steps, :]
output = model(data, send_mask, rec_mask)
print('Tests finished')

# Uncomment if you are using Dataloader class
# train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(args.batch_size, suffix=args.data_suffix, root=False)
# print('Data loader generated')

# for batch_idx, (data, target) in enumerate(test_loader):
#     if args.cuda and torch.cuda.is_available():
#         data.cuda()
#         target.cuda()
#     data = data[:, :, :args.time_steps, :] 
#     output = model(data, send_mask, rec_mask)
#     output = output.view(-1, args.edge_types)
# print('Tests finished')
