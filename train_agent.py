import time
import gym
import numpy as np
from collections import deque
from Utils.preporcessing import preprocess_frame, stack_frame
from Utils.visualizing import plot_score_per_ep, plot_duration_per_ep
from Utils.email_helper import send_mail

def make_env(game, human=False):
    if (human):
        env = gym.make(game, render_mode='human')
    else:
        env = gym.make(game)
    return env

def random_play(env):
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

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (8, -12, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

def train(env, agent, start_epoch, epsilon_by_epsiode, network, n_episodes=1000):
    scores = []
    scores_window = deque(maxlen=20)
    durations = []

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
        scores_window.append(score)       
        scores.append(score)              
        
        end_time = time.time()
        duration = end_time - start_time
        durations.append(duration)

        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}\tDuration: {}s'.format(i_episode, np.mean(scores_window), eps, round(duration, 2)))

    plot_score_per_ep(scores, n_episodes, network)
    plot_duration_per_ep(durations, n_episodes, network)

    start_string = time.strftime("%H:%M:%S", start_time_local)
    end_string = time.strftime("%H:%M:%S", time.localtime())

    send_mail('{} {} episodes training done.\nStart Time: {}\nEnd Time: {}'.format(network, n_episodes, start_string, end_string))

    return agent