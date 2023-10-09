import time
import gym
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import date
from collections import deque
from Utils.preporcessing import preprocess_frame, stack_frame
from Utils.email_helper import send_mail
from Networks.DDQN import DDQNCnn
from Agents.DQN_Agent import DQNAgent

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

env = gym.make('ALE/SpaceInvaders-v5')

# env.seed(0)
# print("The size of frame is: ", env.observation_space.shape)
# print("No. of Actions: ", env.action_space.n)
env.reset()
# plt.figure()
# plt.imshow(env.reset())
# plt.title('Original Frame')
# plt.show()

def random_play():
    score = 0
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            env.close()
            print("Your Score at end of game is: ", score)
            break

# random_play()

# plt.figure()
# plt.imshow(preprocess_frame(env.reset(), (1, -1, -1, 1), 84), cmap="gray")
# plt.title('Pre Processed image')
# plt.show()

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (8, -12, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
GAMMA = 0.99           # discount factor
BUFFER_SIZE = 100000   # replay buffer size
BATCH_SIZE = 64        # Update batch size
LR = 0.0001            # learning rate 
TAU = 1e-3             # for soft update of target parameters
UPDATE_EVERY = 100     # how often to update the network
UPDATE_TARGET = 10000  # After which thershold replay to be started 
EPS_START = 0.99       # starting value of epsilon
EPS_END = 0.01         # Ending value of epsilon
EPS_DECAY = 100        # Rate by which epsilon to be decayed

agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)

# # watch an untrained agent
# state = stack_frames(None, env.reset(), True) 
# for j in range(200):
#     env.render()
#     action = agent.act(state)
#     next_state, reward, done, _ = env.step(action)
#     state = stack_frames(state, next_state, False)
#     if done:
#         break 
        
# env.close()

start_epoch = 0
scores = []
scores_window = deque(maxlen=20)

epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)

# plt.plot([epsilon_by_epsiode(i) for i in range(1000)])

def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    start_time_local = time.localtime()
    
    for i in range(0, 30):
        state = env.reset()
        action = 0
        next_state, reward, done, info = env.step(action)

    for i_episode in range(start_epoch + 1, n_episodes+1):
        start_time = time.time()

        state = stack_frames(None, env.reset(), True)
        score = 0
        eps = epsilon_by_epsiode(i_episode)
        while True:
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            score += reward
            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        
        end_time = time.time()
        duration = end_time - start_time

        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}\tDuration: {}s'.format(i_episode, np.mean(scores_window), eps, round(duration, 2)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('./Stats/{}_DDQN_'.format(n_episodes) + date.today().strftime("%b_%d_%Y") + '.png')
    # plt.show()

    send_mail('DDQN {} episodes training done.\nStart Time: {}\nEnd Time: {}'.format(n_episodes, time.strftime("%H:%M:%S", start_time_local), time.strftime("%H:%M:%S", time.localtime())))

    return scores

# scores = train(1500)

# agent.save_model(1500, True)

agent.load_model(1500, True)

# agent.print_model_state_dict()

print("------------------------------------------------------------------------------------------------------------------------------------")

env = gym.make('ALE/SpaceInvaders-v5', render_mode = 'human')
# env = gym.wrappers.RecordVideo(env, './Videos/')

score = 0
state = stack_frames(None, env.reset(), True)
while True:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    score += reward
    state = stack_frames(state, next_state, False)
    if done:
        print("You Final score is:", score)
        break 
env.close()