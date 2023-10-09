import numpy as np
import matplotlib.pyplot as plt
from datetime import date

def plot_score_per_ep(scores, n_episodes, network):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('./Stats/Scores_{}_{}_'.format(network, n_episodes) + date.today().strftime("%b_%d_%Y") + '.png')

def plot_duration_per_ep(durations, n_episodes, network):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(durations)), durations)
    plt.ylabel('Duration')
    plt.xlabel('Episode #')
    plt.savefig('./Stats/Durations_{}_{}_'.format(network, n_episodes) + date.today().strftime("%b_%d_%Y") + '.png')