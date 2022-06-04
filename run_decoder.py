import time
import torch

import numpy as np
from torch import nn
import argparse
from src.utils import mask, load_data
from src.nri_decoder import MLPDecoder

parser = argparse.ArgumentParser()
# Use this argument to specify a decoder model.
parser.add_argument(
    '--model_path',
    type=str,
    dest='model_path',
    default='saved_model/decoder/20220426_0642/3/',
    help='Specified model path within saved_model/encoder folder')
# Use this argument to specify the corresponding dataset (used in training the model).
parser.add_argument('-ds',
                    '--data_suffix',
                    dest='data_suffix',
                    default='_springsLight5')
# Arguments related to model structures.
parser.add_argument('-sd',
                    '--sep_hidden_dims',
                    dest='sep_hidden_dims',
                    default=256)
parser.add_argument('-so', '--sep_out_dims', dest='sep_out_dims', default=256)
parser.add_argument('-hid', '--hidden_dims', dest='hidden_dims', default=256)
parser.add_argument('-ps', '--time_step', dest='time_steps_test', default=49)
parser.add_argument('-pred_s', '--pred_step', dest='pred_steps', default=1)
parser.add_argument('-dr', '--dropout_rate', dest='dropout', default=0.05)
# Arguments realted to simulation data.
parser.add_argument('-nn', '--num_nodes', dest='num_nodes', default=5)
parser.add_argument('-n', '--node_dims', dest='node_dims', default=4)
parser.add_argument('-et', '--edge_types', dest='edge_types', default=2)
# Arguments realted to training settings.
parser.add_argument('-b', '--batch_size', dest='batch_size', default=5)
# added action to --cuda argument, this will give a bool value rather than a string, and changed the default to false
# with action='store_true', you can use it as a flag, just use --cuda to enable cuda
parser.add_argument('-cuda', '--cuda', dest='cuda', default=False, action='store_true',
                    help='Use this flag if you want to run on GPU')

args = parser.parse_args()

torch.cuda.manual_seed(42)

# get the device based on availability and args flag
device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

send_mask, rec_mask = mask(args.num_nodes)
model = MLPDecoder(args.node_dims, args.sep_hidden_dims, args.sep_out_dims,
                   args.edge_types, args.hidden_dims, args.dropout)
model.load_state_dict(torch.load(args.model_path + 'model.ckpt', map_location='cpu'))

model = model.to(device)

send_mask = send_mask.to(device)
rec_mask = rec_mask.to(device)

if device == 'cuda':
    print('Run in GPU')
else:
    print('No GPU provided.')

train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
    args.batch_size, suffix=args.data_suffix, root=True)

# Tensor for storing results
ret_output = np.zeros([
    len(test_loader), args.batch_size, args.num_nodes,
    args.time_steps_test - 1, args.node_dims
])
ret_target = np.zeros([
    len(test_loader), args.batch_size, args.num_nodes,
    args.time_steps_test - 1, args.node_dims
])

for batch_idx, (input_batch, relations) in enumerate(test_loader):
    input_batch = input_batch.to(device)
    relations = relations.to(device)

    rel_type_onehot = torch.FloatTensor(input_batch.size(0), rec_mask.size(0),
                                        args.edge_types).to(device)
    rel_type_onehot.zero_()
    rel_type_onehot.scatter_(2, relations.view(input_batch.size(0), -1, 1), 1)
    input_batch = input_batch[:, :, -args.time_steps_test:, :]

    target = input_batch[:, :, 1:, :]
    output = model(input_batch, rel_type_onehot, send_mask, rec_mask,
                   args.pred_steps)
    ret_output[batch_idx, :, :, :, :] = output.detach().cpu().numpy()
    ret_target[batch_idx, :, :, :, :] = target.detach().cpu().numpy()

with open(f'saved_results/decoder_result/{args.data_suffix + "output"}.npy',
          'wb') as f:
    np.save(f, ret_output)

with open(f'saved_results/decoder_result/{args.data_suffix + "target"}.npy',
          'wb') as f:
    np.save(f, ret_target)

print('----------------Generated results have been saved-------------------')
