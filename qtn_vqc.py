#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:48:10 2023

@author: junqi
"""

import torchquantum as tq
import torch
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np 
import random 
import torch.nn as nn

from torchquantum.datasets import MNIST 
from torch.optim.lr_scheduler import CosineAnnealingLR 

from torchquantum.encoding import encoder_op_list_name_dict 
from torchquantum.layers import U3CU3Layer0 

from vqc import VQC


class TrainableConvMPS(tq.QuantumModule):
    def __init__(self,
                 n_wires: int = 4,
                 n_qlayers: int = 1):
        super().__init__()
        self.n_wires = n_wires
        self.n_qlayers = n_qlayers 
        
        # Setting up a tensor product encoder
        enc_cnt = list()
        for i in range(self.n_wires):
            cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
            enc_cnt.append(cnt)
        self.encoder = tq.GeneralEncoder(enc_cnt)

        self.arch = {"n_wires": self.n_wires, "n_blocks": 5, "n_layers_per_block": 2}
        self.q_layer = U3CU3Layer0(self.arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
        
    def forward(self, 
                x: torch.tensor, 
                qdev: tq.QuantumDevice, 
                use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 4, 4)
        size = 4
        stride = 2
        x = x.view(bsz, size, size)
        
        data_list = []
        
        for c in range(0, size, stride):
            for r in range(0, size, stride):
                data = torch.transpose(
                    torch.cat(
                        (x[:, c, r], x[:, c, r + 1], x[:, c + 1, r], x[:, c + 1, r + 1])
                    ).view(4, bsz),
                    0,
                    1,
                )
                if use_qiskit:
                    data = self.qiskit_processor.process_parameterized(
                        qdev, self.encoder, self.q_layer, self.measure, data
                    )
                else:
                    self.encoder(qdev, data)
                    self.q_layer(qdev)
                    data = self.measure(qdev)

                data_list.append(data.view(bsz, 4))
                    
        # transpose to (bsz, channel, 2x2)
        result = torch.transpose(
            torch.cat(data_list, dim=1).view(bsz, 4, 4), 1, 2
        ).float()
        
        return result
    

class QFC(tq.QuantumModule):
    def __init__(self,
                 n_wires: int = 4,
                 n_qlayers: int = 1):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict["4x4_ryzxy"])
        self.arch = {"n_wires": self.n_wires, "n_blocks": 4, "n_layers_per_block": 2}
        
        self.q_layer = U3CU3Layer0(self.arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
        
    def forward(self, 
                x: torch.tensor, 
                qdev: tq.QuantumDevice,
                use_qiskit=False):
        data = x
        if use_qiskit:
            data = self.qiskit_processor.process_parameterized(
                qdev, self.encoder, self.q_layer, self.measure, data
            )
        else:
            self.encoder(qdev, data)
            self.q_layer(qdev)
            data = self.measure(qdev)
            
        return data
    
    
def train(dataflow, model, device, optimizer):
    for feed_dict in dataflow["train"]:
        inputs = feed_dict["image"].to(device)
        targets = feed_dict["digit"].to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}\n", end="\r")
        
        
def valid_test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict["image"].to(device)
            targets = feed_dict["digit"].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set loss: {loss}\n")

    return accuracy, loss


class Model(torch.nn.Module):
    def __init__(self,
                 n_wires: int=4, 
                 n_qlayers: int=1,
                 n_class = 3):
        super().__init__()
        self.qf = TrainableConvMPS(n_wires=n_wires, n_qlayers=n_qlayers)
        self.vqc = QFC(n_wires=n_wires, n_qlayers=n_qlayers)
        self.n_wires = n_wires
        self.post_net = nn.Linear(n_wires, n_class)

    def forward(self, x, use_qiskit=False):
        x = x.view(-1, 28, 28)
        bsz = x.shape[0]
        
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        x = self.qf(x, qdev)
        x = x.reshape(-1, 16)
        q_in = torch.sigmoid(x) * np.pi / 2.0
        q_out = self.vqc(q_in, qdev)
        q_class = self.post_net(q_out)
        
        return F.log_softmax(q_class, dim=1)


if __name__ == "__main__":
    n_epochs = 12
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    dataset = MNIST(
        root="./mnist_data",
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[2, 4, 6],
        n_test_samples=300,
        n_train_samples=500,
    )
    
    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=10,
            sampler=sampler,
            num_workers=8,
            pin_memory=True,
        )
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model =  Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}:")
        train(dataflow, model, device, optimizer)
        print(optimizer.param_groups[0]["lr"])
        # valid
        accu, loss = valid_test(dataflow, "test", model, device)
        scheduler.step()
    
        print("Iter: {0}, acc: {1}".format(epoch, accu))
    