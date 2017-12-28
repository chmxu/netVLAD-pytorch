import os
import torch
import torch.utils.data as data
import numpy as np

def default_loader(path):
    return np.fromfile(path, dtype="float32").reshape([-1, 2048])

def transform(fea, max_frames):
    num_frames, _ = fea.shape
    if num_frames >= max_frames:
        #idx = sorted(np.random.choice(num_frames, max_frames))
        #new_fea = np.take(fea, idx, axis=0)
        #idx = np.random.choice(num_frames-max_frames+1, 1)[0]
        new_fea = fea[:max_frames, :]
    else:
        new_fea = np.zeros([max_frames, 2048])
        new_fea[0:num_frames, :] = fea
    return torch.Tensor(new_fea)

class videoDataset(data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None, suffix="_pool5_ucf_senet.binary", loader=default_loader, max_frames=200, sep=","):
        fh = open(label)
        videos = []
        for line in fh.readlines():
            video_id, classes = line.strip().split(sep, 1)
            classes = classes.split(sep)
            videos.append((video_id, list(map(lambda x: int(x)-1, classes))))
        self.root = root
        self.videos = videos
        self.transform = lambda x:transform(x, max_frames)
        self.target_transform = target_transform
        self.loader = loader
        self.suffix = suffix

    def __getitem__(self, index):
        fn, label = self.videos[index]
        fea = self.loader(os.path.join(self.root, fn+self.suffix))
        if self.transform is not None:
            fea = self.transform(fea)
        return fea, torch.LongTensor(label), fn
        # return fea

    def __len__(self):
        return len(self.videos)

# torch.utils.data.DataLoader
if __name__ == "__main__":
    dataset = videoDataset(root="/workspace/untrimmed-data-xcm/UCF-fea-itrc/",
                   label="./ucf_train.txt", transform=transform, sep=' ')
    videoLoader = torch.utils.data.DataLoader(dataset,
                                      batch_size=32, shuffle=True, num_workers=2)

    for i, (features, labels, ids) in enumerate(videoLoader):
        print((labels, ids))
