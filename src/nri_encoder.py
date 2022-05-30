from concurrent.futures import process
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
import math
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """The building block for the MLP-based encoder

    """

    def __init__(self, in_dims, hidden_dims, out_dims, dropout_rate):
        super().__init__()

        self.layer1 = nn.Linear(in_dims, hidden_dims)
        self.elu1 = nn.ELU()

        # During training, randomly zeroes some of the elements of the input tensor with probability p
        self.drop_out = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(hidden_dims, out_dims)
        self.elu2 = nn.ELU()

        # BatchNormalization along the output dimension
        self.bn = nn.BatchNorm1d(out_dims)

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        # reshape back
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, input_batch):
        # Reshape input. The third dimension is #node_dims * #timesteps
        input_batch = input_batch.view(input_batch.size(0),
                                       input_batch.size(1), -1)
        out = self.layer1(input_batch)
        out = self.elu1(out)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.elu2(out)
        return self.batch_norm(out)


class MLPEncoder(nn.Module):

    def __init__(self, in_dims, hidden_dims, out_dims, drop_rate):
        super().__init__()
        self.mlp_block_1 = MLPBlock(in_dims, hidden_dims, hidden_dims,
                                    drop_rate)
        self.mlp_block_2 = MLPBlock(hidden_dims * 2, hidden_dims, hidden_dims,
                                    drop_rate)
        self.out_layer = nn.Linear(hidden_dims, out_dims)
        self.init_weights()

    def init_weights(self):
        """Function for weight initialization
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.1)

    def node2edge(self, input_batch, send_mask, rec_mask):
        """Transferring node representations to edge (sender -> receiver) representations

        Args:
            node_input:
            send_mask: Mask for sender
            rec_mask: Mask for receiver

        Returns:
            Edge representations.
            In the default argument settings, the last 2 dimensions of the tensor are 20 and 2.
        """
        send_batch = torch.matmul(send_mask, input_batch)
        rec_batch = torch.matmul(rec_mask, input_batch)
        return torch.cat([send_batch, rec_batch], dim=2)

    def forward(self, input_batch, send_mask, rec_mask):
        processing_batch = self.mlp_block_1(input_batch)
        # Transfer node to edge. Every edge has a embedded representation (dim=256)
        # `processing_batch` with shape: [#sims, #nodes * (#nodes - 1), hidden_dims * 2]
        processing_batch = self.node2edge(processing_batch, send_mask,
                                          rec_mask)
        # `processing_batch` with shape: [#sims, #nodes * (#nodes - 1), hidden_dims]
        processing_batch = self.mlp_block_2(processing_batch)
        # `output` with shape: [#sims, #nodes * (#nodes - 1), 2]
        output = self.out_layer(processing_batch)

        return output

class CNNBlock(nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims, dropout_rate):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(hidden_dims)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,\
            dilation=1, return_indices=False, ceil_mode=False)
        
        self.conv2 = nn.Conv1d(hidden_dims, out_dims, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_dims)
        
        self.conv_predict = nn.Conv1d(hidden_dims, out_dims, kernel_size=1)
        self.conv_attention = nn.Conv1d(hidden_dims, 1, kernel_size=1)

    def forward(self, input_batch):
        out = self.conv1(input_batch)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.dropout(out)

        out = self.pool(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)

        pred = self.conv_predict(out)
        attention = self.conv_attention(out)
        attention = F.softmax(attention.transpose(1, 0).contiguous())
        attention = attention.transpose(1, 0)

        return (pred * attention).mean(dim=2)

class CNNEncoder(nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims, dropout_rate=0.):
        super().__init__()

        self.cnn = CNNBlock(in_dims*2, hidden_dims, hidden_dims, dropout_rate)
        self.mlp1 = MLPBlock(hidden_dims, hidden_dims, hidden_dims, dropout_rate)
        self.mlp2 = MLPBlock(hidden_dims, hidden_dims, hidden_dims, dropout_rate)
        self.mlp3 = MLPBlock(hidden_dims*3, hidden_dims, hidden_dims, dropout_rate)
        self.fc_out = nn.Linear(hidden_dims, out_dims)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, input_batch, rec_mask):
        rec_batch = rec_mask.t() @ input_batch
        return rec_batch / rec_batch.shape[1]

    def node2edge(self, input_batch, send_mask, rec_mask):
        rec_batch = rec_mask @ input_batch
        send_batch = send_mask @ input_batch
        return torch.cat([send_batch, rec_batch], dim=2)

    def node2edge_init(self, input_batch, send_mask, rec_mask):
        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        input_batch_view = input_batch.view(input_batch.shape[0], input_batch.shape[1], -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        rec_batch = rec_mask @ input_batch_view
        rec_batch = rec_batch.view(input_batch.shape[0] * rec_batch.shape[1],\
            input_batch.shape[2], input_batch.shape[3])
        rec_batch = rec_batch.transpose(2, 1)
        # rec_batch shape: [num_sims*num_edges, num_dims, num_timesteps]

        send_batch = send_mask @ input_batch_view
        send_batch = send_batch.view(input_batch.shape[0] * send_batch.shape[1],\
            input_batch.shape[2], input_batch.shape[3])
        send_batch = send_batch.transpose(2, 1)
        # send_batch shape: [num_sims*num_edges, num_dims, num_timesteps]
        return torch.cat([send_batch, rec_batch], dim=1)


    def forward(self, input_batch, send_mask, rec_mask):
        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_init(input_batch, rec_mask, send_mask)
        processing_batch = self.cnn(edges)
        processing_batch = processing_batch.view(input_batch.shape[0], \
            (input_batch.shape[1]-1) * input_batch.shape[1], -1)
        # New shape: [num_sims, num_edges, num_timesteps*num_dims]
        processing_batch = self.mlp1(processing_batch)
        skip_batch = torch.clone(processing_batch)

        processing_batch = self.edge2node(processing_batch, rec_mask)
        processing_batch = self.mlp2(processing_batch)

        processing_batch = self.node2edge(processing_batch, send_mask, rec_mask)
        processing_batch = torch.cat([processing_batch, skip_batch], dim=2)
        processing_batch = self.mlp3(processing_batch)

        return self.fc_out(processing_batch)
        