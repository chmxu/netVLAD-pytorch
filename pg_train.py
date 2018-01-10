import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from loss import CrossEntropy
from torch.distributions import Bernoulli
from dataset import videoDataset, transform
from netVLAD import NetVLADModelLF
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

class Classifier(nn.Module):
    def __init__(self, feature_size, num_classes, hidden_size=1024):
        super(Classifier, self).__init__()
        self.input_size = feature_size
        self.output_size = num_classes
	self.bn = nn.BatchNorm1d(hidden_size, eps=1e-3, momentum=0.01)
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.output_size)

    def forward(self, model_input):
        fc1 = F.relu6(self.bn(self.fc1(model_input)))
        return F.sigmoid(self.fc2(fc1))

class policyNet(nn.Module):
    def __init__(self, max_frames, feature_size, num_hist):
        super(policyNet, self).__init__()
        self.input_size = feature_size + num_hist
        self.fc1 = nn.Linear(self.input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)
        self.max_frames = max_frames
        self.num_hist = num_hist

    def forward(self, model_input):
        history_vector = Variable(torch.zeros([model_input.shape[0], self.num_hist])).cuda()
        actions = []
        log_probs = []
        for t in range(self.max_frames):
            video_frames = model_input[:, t, :]
            input = torch.cat([video_frames, history_vector], dim=1)
            fc1 = F.relu(self.fc1(input))
            fc2 = F.relu(self.fc2(fc1))
            dists = F.softmax(self.fc3(fc2), dim=1)[:, 0]  # (batch_size, 1)
            m = Bernoulli(dists)
            action = m.sample()
            actions.append(action.view([-1, 1]))
            log_probs.append(m.log_prob(action).view([-1, 1]))
            history_vector = torch.cat([history_vector[:, 1:], action], dim=1)
        return torch.cat(actions, dim=1), torch.cat(log_probs, dim=1)

class Aggregate(nn.Module):
    def __init__(self, classifier, policy_net, max_frames, feature_size, num_classes, num_hist=10, add_bn=True, policy=True):
        super(Aggregate, self).__init__()
        self.classifier = classifier(feature_size, num_classes)
        self.policy_net = policy_net(max_frames, feature_size, num_hist)
        if torch.cuda.is_available():
            self.classifier.cuda()
            self.policy_net.cuda()
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.batch_norm_input = nn.BatchNorm1d(feature_size, eps=1e-3, momentum=0.01)
        self.add_bn = add_bn
	self.policy = policy

    def take_action(self, input, action):
        reshaped_input = input.view([-1, self.feature_size])
        action = action.view([-1, 1])
        distilled_frames = action * reshaped_input
        distilled_frames = distilled_frames.view([input.shape[0], -1, self.feature_size])
	return distilled_frames.max(1)[0]

    def forward(self, model_input):
        reshaped_input = model_input.view([-1, self.feature_size])
        if self.add_bn:
            reshaped_input = self.batch_norm_input(reshaped_input)
        input = reshaped_input.view([model_input.shape[0], -1, self.feature_size])
	if self.policy:
            action, log_prob = self.policy_net(input)
	    deleted = torch.sum(action == 0, dim=1).float() / action.shape[1]
            output = self.take_action(input, action)
        else:
	    output, _ = torch.max(input, dim=1)
	    log_prob = None
	    deleted = None
        logits = self.classifier(output)
        return logits, log_prob, deleted

def loss_fn(num_classes, logits, labels):
    return CrossEntropy(num_classes=num_classes)(logits, labels)

def accuracy(predictions, actuals):
    top_prediction = torch.max(predictions, 1)[1]
    #hits = actuals[torch.arange(actuals.shape[0]), top_prediction]
    return torch.mean((top_prediction.view([-1, 1])==actuals.view([-1, 1])).float())

def train(num_epochs):
    dataset = videoDataset(root="/workspace/untrimmed-data-xcm/UCF-fea-itrc/",
                           label="./labels/UCF/ucf_train.txt", transform=transform, sep=" ", max_frames=200)
    videoLoader = torch.utils.data.DataLoader(dataset,
                                              batch_size=80, shuffle=True, num_workers=0)
    valset = videoDataset(root="/workspace/untrimmed-data-xcm/UCF-fea-itrc",
                   label="./labels/UCF/ucf_test.txt", transform=transform, sep=" ", max_frames=200)
    valLoader = torch.utils.data.DataLoader(valset,
                                      batch_size=80, shuffle=False, num_workers=0)

    aggregate = Aggregate(classifier=Classifier,
                        policy_net=policyNet,
                        max_frames=200,
                        feature_size=2048,
                        num_classes=101)

    if torch.cuda.is_available():
        aggregate.cuda()
    aggregate.load_state_dict(torch.load("./models/pg/aggregate_epoch24.pt"))
    for parameter in aggregate.classifier.parameters():
        parameter.requires_grad = False
    for param in aggregate.batch_norm_input.parameters():
	param.requires_grad = False
    optimizer = optim.Adam(params=aggregate.policy_net.parameters(), lr=1e-5)
    for epoch in range(num_epochs):
        for i, (features, labels, ids) in enumerate(videoLoader):
            if torch.cuda.is_available():
                features = Variable(features).cuda()
                labels = Variable(labels).cuda()
            logits, log_probs, deleted = aggregate(features)
            reward = loss_fn(101, logits, labels) + 5 * torch.mean(deleted) if deleted is not None else loss_fn(101, logits, labels)
            loss = torch.mean(torch.sum(-reward * log_probs, dim=1)) if log_probs is not None else ceLoss
            acc = accuracy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch: " + str(epoch) + " Iter: " + str(i) + " Acc: " + ("%.2f" % acc) +
                  " Loss: " + str(loss.data[0]))
        val_acc = 0
        val_sample = 0
        aggregate.eval()
        for _, (val_features, val_labels, _) in enumerate(valLoader):
            if torch.cuda.is_available():
                val_features = Variable(val_features).cuda(0)
                val_labels = Variable(val_labels).cuda(0)
            logits, log_probs, deleted = aggregate(val_features)
            val_acc += accuracy(logits, val_labels) * val_labels.shape[0]
            val_sample += val_labels.shape[0]
	aggregate.train()
        #print("%d val samples have done" % val_sample)
        #if total_sample > 2000:
        #    break
        print("Epoch " + str(epoch) + " Val Acc: " + ("%.3f" % (val_acc/val_sample)))
	torch.save(aggregate.state_dict(), "./models/pg/" + "aggregate_epoch%d.pt" % epoch)
if __name__ == "__main__":
    train(50)
