import random
import torch.optim as optim
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
USE_CUDA = torch.cuda.is_available()
from policy import QNet
from schedule import LinearSchedule
from dataset import videoDataset, transform
import torch
import torch.utils.data as data

def dqn_learing(
    dataLoader,
    q_func,
    exploration,
    feature_size,
    num_classes,
    r_p,
    replay_buffer_size=1000000,
    num_epochs=50,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000
    ):

    ###############
    # BUILD MODEL #
    ###############
    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t, weights):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            #obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            q_val = model(Variable(obs.data.cpu(), volatile=True).cuda()).data.cpu()
	    if model.training == True:
                weights.append(1.0 - q_val[-1] / torch.sum(q_val))
	    else:
		weights.append(1.0)
            return q_val.max(0)[1].numpy()[0], weights
        else:
            weights.append(1.0)
            return torch.IntTensor([[random.randrange(num_classes+1)]]).numpy()[0], weights

    # Initialize target q function and q function
    Q = q_func(feature_size, num_classes)
    target_Q = q_func(feature_size, num_classes)
    if USE_CUDA:
        Q.cuda()
        target_Q.cuda()
    # Construct Q network optimizer function
    optimizer = optim.Adam(Q.parameters(), lr=1e-4)

    # Construct the replay buffer
    #replay_buffer = [[] for _ in range(num_classes)]
    #category_t = [0 for _ in range(num_classes)]
    replay_buffer = []
    t = 0
    num_param_updates = 0
    pop_index = 0
    for e in range(num_epochs):
        for idx, (video, label, id) in enumerate(dataLoader):
            total_rewards = 0
            category = label.numpy()[0][0]
            video = Variable(video[0])
            weights = []
            historical_feature = Variable(torch.zeros(feature_size))
            last_frame = Variable(torch.zeros(feature_size))
            for j in range(video.shape[0]):
                #category_t[category] += 1
		t += 1
                frame_feature = video[j]
                if j > 0:
                    historical_feature = torch.max(torch.cat([historical_feature.view([-1, 1]), last_frame.view([-1, 1])], dim=1), dim=1)[0]
                curr_state = torch.cat([frame_feature, historical_feature])
                # Choose random action if not yet start learning
                #if category_t[category] > learning_starts:
		if t > learning_starts:
                    #action, weights = select_epilson_greedy_action(Q, curr_state, category_t[category], weights)
		    action, weights = select_epilson_greedy_action(Q, curr_state, t, weights)
                else:
                    weights.append(0.0)
                    action = random.randrange(num_classes+1)
                # Advance one step
                if action < num_classes:
                    done = True
                    reward = 1.0 if action == category else -1.0
                else:
                    if j >= 100:
			done = True
			reward = -2.0
		    else:
                        done = False
                        reward = r_p
                total_rewards += reward
                # Store other info in replay memory
                last_frame = frame_feature
                next_state = torch.cat([video[j+1], torch.max(torch.cat([historical_feature.view([-1, 1]), last_frame.view([-1, 1])], dim=1), dim=1)[0]])
		'''
                if len(replay_buffer[category]) == replay_buffer_size:
                    replay_buffer[category].pop(0)
                replay_buffer[category].append((curr_state, action, reward, next_state, done))
		'''
		if len(replay_buffer) == replay_buffer_size:
		    replay_buffer.pop(pop_index)
    		    replay_buffer.insert(pop_index, (curr_state, action, reward, next_state, done))
		    pop_index = (pop_index + 1) % replay_buffer_size
		else:
		    replay_buffer.append((curr_state, action, reward, next_state, done))
		
                ### Perform experience replay and train the network.
                #if (category_t[category] > learning_starts and
                #        category_t[category] % learning_freq == 0 and
                #        len(replay_buffer[category]) == replay_buffer_size):
		if (t > learning_starts and
			t % learning_freq == 0 and
			len(replay_buffer) == replay_buffer_size):
		    print("Updating policy...")
                    # Use the replay buffer to sample a batch of transitions
                    # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
                    # in which case there is no Q-value at the next state; at the end of a
                    # episode, only the current state reward contributes to the target
                    #samples = random.sample(replay_buffer[category], batch_size)
		    samples = random.sample(replay_buffer, batch_size)
                    obs_batch = torch.cat(list(map(lambda x:x[0].view([1, -1]), samples)), 0)
		    act_batch = list(map(lambda x:x[1], samples))
		    rew_batch = list(map(lambda x:x[2], samples))
		    next_obs_batch = torch.cat(list(map(lambda x:x[3].view([1, -1]), samples)), 0)
		    done_mask = list(map(lambda x:x[4], samples))
                    # Convert numpy nd_array to torch variables for calculation
                    act_batch = Variable(torch.from_numpy(np.array(act_batch, dtype="int32")).long())
                    rew_batch = Variable(torch.Tensor(rew_batch))
                    not_done_mask = Variable(1 - torch.Tensor(done_mask))

                    if USE_CUDA:
			obs_batch = obs_batch.cuda()
                        act_batch = act_batch.cuda()
                        rew_batch = rew_batch.cuda()
		        next_obs_batch = next_obs_batch.cuda()
			not_done_mask = not_done_mask.cuda()

                    # Compute current Q value, q_func takes only state and output value for every state-action pair
                    # We choose Q based on action taken.
                    current_Q_values = Q(obs_batch).gather(1, act_batch.view([-1, 1]))
                    # Compute next Q value based on which action gives max Q values
                    # Detach variable from the current graph since we don't want gradients for next Q to propagated
                    next_max_q = target_Q(next_obs_batch).detach().max(1)[0]
                    next_Q_values = not_done_mask * next_max_q
                    # Compute the target of the current Q values
                    target_Q_values = rew_batch + (gamma * next_Q_values)
                    # Compute Bellman error
                    bellman_error = F.smooth_l1_loss(current_Q_values, target_Q_values)
                    optimizer.zero_grad()
                    # run backward pass
                    #current_Q_values.backward(d_error)
	            bellman_error.backward()
                    for param in Q.parameters():
                        param.grad.data.clamp_(-1, 1)
                    # Perfom the update
                    optimizer.step()
                    num_param_updates += 1

                    # Periodically update the target network by Q network to target Q network
                    if num_param_updates % target_update_freq == 0:
                        target_Q.load_state_dict(Q.state_dict())

                if done == True:
                    break
            print("Epoch %d, video: %d, total reward: %.4f, randomness: %.3f" % (e, idx, total_rewards, exploration.value(t)))
        if e > 20:
            torch.save(Q.state_dict(), './models/qlearning/QNet_epoch%d.pt' % e)


GAMMA = 0.9
REPLAY_BUFFER_SIZE = 50000
LEARNING_STARTS = 5000
LEARNING_FREQ = 2
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 2000
BATCH_SIZE = 32
dataset = videoDataset(root="/workspace/untrimmed-data-xcm/UCF-fea-itrc/",
                   label="./labels/UCF/ucf_train.txt", transform=transform, sep=' ', max_frames=300)
videoLoader = torch.utils.data.DataLoader(dataset,
                                      batch_size=1, shuffle=True, num_workers=4)

def main(num_epochs):

    exploration_schedule = LinearSchedule(300000, 0.1)

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

if __name__ == "__main__":
    main(50)
