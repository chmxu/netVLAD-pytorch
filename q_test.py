import random
import torch.optim as optim
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from dataset import videoDataset, transform
from policy import QNet
import time
USE_CUDA = torch.cuda.is_available()
#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

"""
class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        super(Variable, self).__init__(data, *args, **kwargs)
        if USE_CUDA:
            data = data.cuda()
"""

def dqn_learing(
    dataLoader,
    q_func,
    feature_size,
    num_classes,
    model_pt,
    w
    ):

    Q = q_func(feature_size, num_classes)
    if USE_CUDA:
        Q.cuda()
    Q.load_state_dict(torch.load(model_pt))
    for idx, (video, label, id) in enumerate(dataLoader):
	done = False
        if id[0] == "v_PlayingFlute_g05_c01":
	    import pdb;pdb.set_trace()
        print(id[0])
	start = time.time()
        total_rewards = 0
        category = label.numpy()[0][0]
        video = Variable(video[0])
        weights = []
        historical_feature = Variable(torch.zeros(feature_size))
        last_frame = Variable(torch.zeros(feature_size))
        for j in range(video.shape[0]):
            frame_feature = video[j]
            if j > 0:
    	        historical_feature = torch.max(torch.cat([historical_feature.view([-1, 1]), last_frame.view([-1, 1])], dim=1), dim=1)[0]
            curr_state = torch.cat([frame_feature, historical_feature])
            q_val = Q(Variable(curr_state.data.cpu(), volatile=True).cuda()).data.cpu()
	    action = q_val.max(0)[1].numpy()[0]
            if action < num_classes:
		done = True
                w.write(id[0] + ' ' + str(action+1) + ' ' + str(j+1)  +'\n')		
		print(id[0] + ' ' + str(action+1) + ' ' + str(j+1))
		break
	    last_frame = frame_feature
	if done == False:
    	    action = q_val[:-1].max(0)[1].numpy()[0]
	    w.write(id[0] + ' ' + str(action+1) + ' ' + str(j+1)  +'\n')
	if idx % 500 == 0:
	    print("%d samples have done" % idx)


dataset = videoDataset(root="/workspace/untrimmed-data-xcm/UCF-fea-itrc/",
                   label="./labels/UCF/ucf_test.txt", transform=transform, sep=' ', max_frames=250)
videoLoader = torch.utils.data.DataLoader(dataset,
                                      batch_size=1, shuffle=False, num_workers=0)

def main():
    w = open("./script/result/result.txt", 'w')
    dqn_learing(
        dataLoader=videoLoader,
        feature_size=2048,
        num_classes=101,
        q_func=QNet,
	model_pt="./models/qlearning/QNet_epoch44.pt",
	w=w
    )
    w.close()

if __name__ == "__main__":
    main()
