"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import random
import torch.optim as optim
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
USE_CUDA = torch.cuda.is_available()
#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

"""
class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)
"""

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
            q_val = model(Variable(obs.data.cpu(), volatile=True)).data.cpu()
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
    #if USE_CUDA:
    #    Q.cuda()
    #    target_Q.cuda()
    # Construct Q network optimizer function
    optimizer = optim.Adam(Q.parameters(), lr=1e-4)

    # Construct the replay buffer
    replay_buffer = [[] for _ in range(num_classes)]
    category_t = [0 for _ in range(num_classes)]

    num_param_updates = 0
    for e in range(num_epochs):
        for idx, (video, label, id) in enumerate(dataLoader):
            total_rewards = 0
            category = label.numpy()[0][0]
            video = Variable(video[0])
            weights = []
            historical_feature = Variable(torch.zeros(feature_size))
            last_frame = Variable(torch.zeros(feature_size))
            for j in range(video.shape[0]):
                category_t[category] += 1
                frame_feature = video[j]
                if j > 0:
                    historical_feature = (historical_feature * (idx-1) + last_frame) / idx
                curr_state = torch.cat([frame_feature, historical_feature])
                # Choose random action if not yet start learning
                if category_t[category] > learning_starts:
                    action, weights = select_epilson_greedy_action(Q, curr_state, category_t[category], weights)
                else:
                    weights.append(0.0)
                    action = random.randrange(num_classes+1)
                # Advance one step
                if action < num_classes:
                    done = True
                    reward = 1.0 if action == category else -1.0
                else:
                    done = False
                    reward = r_p
                total_rewards += reward
                # Store other info in replay memory
                last_frame = frame_feature
                next_state = torch.cat([video[j+1], (historical_feature * idx + last_frame) / (idx + 1)])
                if len(replay_buffer[category]) == replay_buffer_size:
                    replay_buffer[category].pop(0)
                replay_buffer[category].append((curr_state, action, reward, next_state, done))

                ### Perform experience replay and train the network.
                if (category_t[category] > learning_starts and
                        category_t[category] % learning_freq == 0 and
                        len(replay_buffer[category]) == replay_buffer_size):
		    print("Updating policy...")
                    # Use the replay buffer to sample a batch of transitions
                    # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
                    # in which case there is no Q-value at the next state; at the end of a
                    # episode, only the current state reward contributes to the target
                    samples = random.sample(replay_buffer[category], batch_size)
                    obs_batch = torch.cat(list(map(lambda x:x[0].view([1, -1]), samples)), 0)
		    act_batch = list(map(lambda x:x[1], samples))
		    rew_batch = list(map(lambda x:x[2], samples))
		    next_obs_batch = torch.cat(list(map(lambda x:x[3].view([1, -1]), samples)), 0)
		    done_mask = list(map(lambda x:x[4], samples))
                    # Convert numpy nd_array to torch variables for calculation
                    act_batch = Variable(torch.from_numpy(np.array(act_batch, dtype="int32")).long())
                    rew_batch = Variable(torch.Tensor(rew_batch))
                    not_done_mask = Variable(1 - torch.Tensor(done_mask))

                    #if USE_CUDA:
                    #    act_batch = act_batch.cuda()
                    #    rew_batch = rew_batch.cuda()

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
            print("Epoch %d, video: %d, total reward: %.4f" % (e, idx, total_rewards))


