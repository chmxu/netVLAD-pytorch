import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable
from loss import CrossEntropy

class selfAttn(nn.Module):
    def __init__(self, feature_size, time_step, hidden_size, num_desc):
        super(selfAttn, self).__init__()
        self.linear_1 = nn.Linear(feature_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, feature_size)
        self.num_desc = num_desc
        #self.init_weights()
    
    def init_weights(self):
        self.linear_1.weight.data.uniform_(-0.1, 0.1)
        #self.linear_2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, model_input):   # (batch_size, time_step, feature_size)
        #reshaped_input = model_input.permute(0, 2, 1)  # (batch_size, feature_step, time_step)
        #s1 = F.tanh(self.linear_1(reshaped_input))  # (batch_size, feature_size, hidden_size)
        s1 = F.relu(self.linear_1(model_input))  # (batch_size, feature_size, num_desc)
        s2 = F.sigmoid(self.linear_2(s1))
        '''
        M = t  # (batch_size, time_step, num_desc)
        I = Variable(torch.eye(self.num_desc)).cuda()
        AAT = torch.matmul(A.permute(0, 2, 1), A)
        #import pdb;pdb.set_trace()
        P = torch.norm(AAT - I, 2)
        penal = P * P / model_input.shape[0]
        '''
        output = model_input * s2
        return output

class NetVLAD(nn.Module):
    def __init__(self, feature_size, max_frames,cluster_size, add_bn=False, truncate=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size / 2 if truncate else feature_size
        self.max_frames = max_frames
        self.cluster_size = cluster_size
        self.batch_norm = nn.BatchNorm1d(cluster_size, eps=1e-3, momentum=0.01)
        self.linear = nn.Linear(self.feature_size, self.cluster_size)
        self.softmax = nn.Softmax(dim=1)
        self.cluster_weights2 = nn.Parameter(torch.FloatTensor(1, self.feature_size,
                                                               self.cluster_size))
        self.add_bn = add_bn
        self.truncate = truncate
        self.first = True
        self.init_parameters()

    def init_parameters(self):
        init.normal(self.cluster_weights2, std=1 / math.sqrt(self.feature_size))

    def forward(self, reshaped_input):
        random_idx = torch.bernoulli(torch.Tensor([0.5]))
        if self.truncate:
            if self.training == True:
                reshaped_input = reshaped_input[:, :self.feature_size].contiguous() if random_idx[0]==0 else reshaped_input[:, self.feature_size:].contiguous()
            else:
                if self.first == True:
                    reshaped_input = reshaped_input[:, :self.feature_size].contiguous()
                else:
                    reshaped_input = reshaped_input[:, self.feature_size:].contiguous()
        activation = self.linear(reshaped_input)
        if self.add_bn:
            activation = self.batch_norm(activation)
        activation = self.softmax(activation).view([-1, self.max_frames, self.cluster_size])
        a_sum = activation.sum(-2).unsqueeze(1)
        a = torch.mul(a_sum, self.cluster_weights2)
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = reshaped_input.view([-1, self.max_frames, self.feature_size])
        vlad = torch.matmul(activation, reshaped_input).permute(0, 2, 1).contiguous()
        vlad = vlad.sub(a).view([-1, self.cluster_size * self.feature_size])
        if self.training == False:
            self.first = 1 - self.first
        return vlad

class MoeModel(nn.Module):
    def __init__(self, num_classes, feature_size, num_mixture=2):
        super(MoeModel, self).__init__()
        self.gating = nn.Linear(feature_size, num_classes * (num_mixture+1))
        self.expert = nn.Linear(feature_size, num_classes * num_mixture)
        self.num_mixture = num_mixture
        self.num_classes = num_classes
    def forward(self, model_input):
        gate_activations = self.gating(model_input)
        gate_dist = nn.Softmax(dim=1)(gate_activations.view([-1, self.num_mixture + 1]))
        expert_activations = self.expert(model_input)
        expert_dist = nn.Softmax(dim=1)(expert_activations.view([-1, self.num_mixture]))
        probabilities_by_class_and_batch = torch.sum(
            gate_dist[:, :self.num_mixture] * expert_dist, 1)
        return probabilities_by_class_and_batch.view([-1, self.num_classes])

class NetVLADModelLF(nn.Module):
    def __init__(self, cluster_size, max_frames, feature_size, hidden_size, num_classes, add_bn=False, use_moe=True, truncate=True, attention=True):
        super(NetVLADModelLF, self).__init__()
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.cluster_size = cluster_size
        self.video_NetVLAD = NetVLAD(self.feature_size, self.max_frames, self.cluster_size, truncate=truncate, add_bn=add_bn) 
        self.batch_norm_input = nn.BatchNorm1d(feature_size, eps=1e-3, momentum=0.01)
        self.batch_norm_activ = nn.BatchNorm1d(hidden_size, eps=1e-3, momentum=0.01)
        self.linear_1 = nn.Linear(cluster_size * self.feature_size / 2, hidden_size) if truncate else nn.Linear(cluster_size * self.feature_size, hidden_size)
        self.relu = nn.ReLU6()
        self.linear_2 = nn.Linear(hidden_size, num_classes)
        self.s = nn.Sigmoid()
        self.moe = MoeModel(num_classes, hidden_size)
	self.Attn = selfAttn(feature_size, max_frames, 128, 64)
        self.add_bn = add_bn
	self.truncate = truncate
        self.use_moe = use_moe
	self.attention = attention
    def forward(self, model_input):
        reshaped_input = model_input.view([-1, 2048])
        if self.add_bn:
            reshaped_input = self.batch_norm_input(reshaped_input)
        if self.attention:
            model_input = self.Attn(reshaped_input)
        vlad = self.video_NetVLAD(reshaped_input)
        if self.add_bn:
            activation = self.batch_norm_activ(self.linear_1(vlad))
        activation = self.relu(activation)
        if self.use_moe:
            logits = self.moe(activation)
        else:
            logits = self.s(self.linear_2(activation))
        return logits

def loss_fn(num_classes, logits, labels):
    return CrossEntropy(num_classes=num_classes)(logits, labels)
 
def accuracy(predictions, actuals):
    top_prediction = np.argmax(predictions, 1)
    hits = actuals[np.arange(actuals.shape[0]), top_prediction]
    return np.average(hits)


