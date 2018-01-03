from q_train import dqn_learing
from policy import QNet
from schedule import LinearSchedule
from dataset import videoDataset, transform
import torch
import torch.utils.data as data
BATCH_SIZE = 200
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 500
LEARNING_STARTS = 100
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 2000

dataset = videoDataset(root="/workspace/untrimmed-data-xcm/UCF-fea-itrc/",
                   label="./labels/UCF/ucf_train.txt", transform=transform, sep=' ', max_frames=250)
videoLoader = torch.utils.data.DataLoader(dataset,
                                      batch_size=1, shuffle=True, num_workers=0)

def main(num_epochs):

    exploration_schedule = LinearSchedule(800, 0.1)

    dqn_learing(
        dataLoader=videoLoader,
        num_epochs=num_epochs,
        feature_size=2048,
        num_classes=101,
        r_p=0.01,
        q_func=QNet,
        exploration=exploration_schedule,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
    )

if __name__ == '__main__':
    main(50)

