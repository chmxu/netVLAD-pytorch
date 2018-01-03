from dataset import videoDataset, transform   
from netVLAD import NetVLADModelLF, loss_fn, accuracy
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import argparse
import torch.nn as nn
parser = argparse.ArgumentParser()
parser.add_argument("--cluster_size", help="cluster size", type=int)
parser.add_argument("--batch_size", help="batch size", type=int)
parser.add_argument("--max_frames", help="max frames", type=int)
parser.add_argument("--feature_size", help="feature size", type=int)
parser.add_argument("--hidden_size", help="hidden size", type=int)
parser.add_argument("--num_classes", help="num classes", type=int)
parser.add_argument("--epoch_num", help="epoch num", type=int)
parser.add_argument("--learning_rate", help="learning_rate", type=float)
parser.add_argument("--l2_regularizer", help="regularizer coefficient", type=float)
parser.add_argument("--continue_train", help="continue from former model", type=bool, default=False)
parser.add_argument("--truncate", help="continue from former model", type=bool, default=False)
parser.add_argument("--save_model", help="save model former", default=None)
parser.add_argument("--root", help="root")
parser.add_argument("--label", help="label")
parser.add_argument("--val_label", help="val label")
parser.add_argument("--sep", help="seperate")
parser.add_argument("--save_dir", help="save model direc")
args = parser.parse_args()
dataset = videoDataset(root=args.root,
                   label=args.label, transform=transform, sep=args.sep, max_frames=args.max_frames)
videoLoader = torch.utils.data.DataLoader(dataset,
                                      batch_size=args.batch_size, shuffle=True, num_workers=0)
valset = videoDataset(root=args.root,
                   label=args.val_label, transform=transform, sep=args.sep, max_frames=args.max_frames)
valLoader = torch.utils.data.DataLoader(valset,
                                      batch_size=args.batch_size, shuffle=True, num_workers=0)
NetVLAD = NetVLADModelLF(cluster_size=args.cluster_size,
                         max_frames=args.max_frames,
                         feature_size=args.feature_size,
                         hidden_size=args.hidden_size,
                         num_classes=args.num_classes,
                         add_bn=True,
                         truncate=args.truncate,
                         use_moe=False)

if torch.cuda.is_available():
    NetVLAD.cuda()
    #NetVLAD = nn.DataParallel(NetVLAD, device_ids=[0, 1])
print(NetVLAD)
NetVLAD.train(mode=True)
if args.continue_train and args.save_model is not None:
    NetVLAD.load_state_dict(torch.load(args.save_model))
optimizer = optim.Adam(params=NetVLAD.parameters(), lr=args.learning_rate, weight_decay=args.l2_regularizer)
#penal_optim = optim.Adam(params=NetVLAD.parameters(), lr=args.learning_rate * 100, weight_decay=args.l2_regularizer)
for epoch in range(args.epoch_num):
    total_loss = 0
    total_sample = 0
    for i, (features, labels, ids) in enumerate(videoLoader):
        if torch.cuda.is_available():
            features = Variable(features).cuda(0)
            labels = Variable(labels).view(-1).cuda(0)
        logits = NetVLAD(features)
        float_labels = torch.zeros(labels.shape[0], args.num_classes).scatter_(1, labels.data.cpu().view(-1, 1), 1.0).numpy()
        classi_loss = loss_fn(args.num_classes, logits, labels)
        loss = classi_loss
        acc = accuracy(logits.data.cpu().numpy(), float_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch: " + str(epoch) + " Iter: " + str(i) + " Acc: " + ("%.2f" % acc) +
              " Classification Loss: " + str(classi_loss.data[0]))
        total_loss += loss.data[0] * labels.shape[0]
        total_sample += labels.shape[0]
    print("Epoch: " + str(epoch) + " Average loss: " + str(total_loss / total_sample))
    val_acc = 0
    val_sample = 0
    NetVLAD.eval()
    for _, (val_features, val_labels, _) in enumerate(valLoader):
        if torch.cuda.is_available():
            val_features = Variable(val_features).cuda(0)
            val_labels = Variable(val_labels).cuda(0)
        float_val_labels = torch.zeros(val_labels.shape[0], args.num_classes).scatter_(1, val_labels.data.cpu().view(-1, 1), 1.0).numpy()
        if args.truncate == False:
            logits = NetVLAD(val_features)
        else:
            logits = (NetVLAD(val_features) + NetVLAD(val_features)) / 2
        val_acc += accuracy(logits.data.cpu().numpy(), float_val_labels) * val_labels.shape[0]
        val_sample += val_labels.shape[0]
        #print("%d val samples have done" % val_sample)
        #if total_sample > 2000:
        #    break
    print("Epoch " + str(epoch) + " Val Acc: " + ("%.3f" % (val_acc/val_sample)))
    NetVLAD.train(mode=True)
    torch.save(NetVLAD.state_dict(), args.save_dir + "NetVLAD_epoch%d.pt" % epoch)
