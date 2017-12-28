from dataset import videoDataset, transform
from netVLAD import NetVLADModelLF
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cluster_size", help="cluster size", type=int)
parser.add_argument("--batch_size", help="batch size", type=int)
parser.add_argument("--max_frames", help="max frames", type=int)
parser.add_argument("--feature_size", help="feature size", type=int)
parser.add_argument("--hidden_size", help="hidden size", type=int)
parser.add_argument("--num_classes", help="num classes", type=int)
parser.add_argument("--epoch_num", help="epoch num", type=int)
parser.add_argument("--learning_rate", help="learning_rate", type=float)
parser.add_argument("--root", help="root")
parser.add_argument("--truncate", help="continue from former model", type=bool, default=False)
parser.add_argument("--label", help="label")
parser.add_argument("--sep", help="seperate")
parser.add_argument("--save_model", help="save model direc")
args = parser.parse_args()
dataset = videoDataset(root=args.root,
                   label=args.label, transform=transform, sep=args.sep, max_frames=args.max_frames)
videoLoader = torch.utils.data.DataLoader(dataset,
                                      batch_size=args.batch_size, shuffle=False, num_workers=0)

NetVLAD = NetVLADModelLF(cluster_size=args.cluster_size,
                         max_frames=args.max_frames,
                         feature_size=args.feature_size,
                         hidden_size=args.hidden_size,
                         num_classes=args.num_classes,
                         add_bn=True,
			 use_moe=False,
                         truncate=args.truncate)

if torch.cuda.is_available():
    NetVLAD.cuda()
    #NetVLAD = nn.DataParallel(NetVLAD, device_ids=[0, 1])
NetVLAD.eval()
NetVLAD.load_state_dict(torch.load(args.save_model))
w = open("./result/test_result.txt", 'w')
total_sample = 0
for i, (features, labels, ids) in enumerate(videoLoader):
    if torch.cuda.is_available():
        features = Variable(features).cuda(0)
        labels = Variable(labels).cuda(0)
    logits = NetVLAD(features)
    for j in range(len(ids)):
        pred = np.argmax(logits.data.cpu()[j, :])
        w.write(ids[j] + ' ' + str(pred) + '\n')
    total_sample += len(ids)
    print("%d samples have done" % total_sample)
w.close()
