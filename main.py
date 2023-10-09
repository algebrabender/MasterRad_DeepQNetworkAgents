import sys
import torch
import math
import train_agent
from test_agent import play
from Networks.DQN import DQNCnn
from Networks.DDQN import DDQNCnn
from Agents.DQN_Agent import DQNAgent

train = sys.argv[1] == 'train' if len(sys.argv) > 1 else True 
test = sys.argv[1] == 'test' if len(sys.argv) > 1 else True
episodes = int(sys.argv[2]) if (len(sys.argv) > 2 and sys.argv[2].isnumeric()) else 1500
agent = sys.argv[3] if len(sys.argv) > 3 else 'all'

#################################################################################################################################################################

INPUT_SHAPE = (4, 84, 84)
GAMMA = 0.99           
BUFFER_SIZE = 100000   
BATCH_SIZE = 64        
LR = 0.0001            
TAU = 1e-3             
UPDATE_EVERY = 100     
UPDATE_TARGET = 10000  
EPS_START = 0.99       
EPS_END = 0.01         
EPS_DECAY = 100        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

#################################################################################################################################################################

if (train):
    env_v4 = train_agent.make_env('SpaceInvaders-v4')
    env_Det_v4 = train_agent.make_env('SpaceInvadersDeterministic-v4')
    env_NoFS_v4 = train_agent.make_env('SpaceInvadersNoFrameskip-v4')
    env_v5 = train_agent.make_env('ALE/SpaceInvaders-v5')

    DQN_agent_v4 = DQNAgent(INPUT_SHAPE, env_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
    DQN_agent_Det_v4 = DQNAgent(INPUT_SHAPE, env_Det_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
    DQN_agent_NoFS_v4 = DQNAgent(INPUT_SHAPE, env_NoFS_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
    DQN_agent_v5 = DQNAgent(INPUT_SHAPE, env_v5.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)

    DDQN_agent_v4 = DQNAgent(INPUT_SHAPE, env_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)
    DDQN_agent_Det_v4 = DQNAgent(INPUT_SHAPE, env_Det_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)
    DDQN_agent_NoFS_v4 = DQNAgent(INPUT_SHAPE, env_NoFS_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)
    DDQN_agent_v5 = DQNAgent(INPUT_SHAPE, env_v5.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)

    start_epoch = 0
    epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)

    #################################################################################################################################################################

    if (agent == 'all'):

        DQN_agent_v4 = train_agent.train(env_v4, DQN_agent_v4, start_epoch, epsilon_by_epsiode, 'DQN_agent_v4', episodes)
        DQN_agent_v4.save_model(episodes, 'DQN_agent_v4')
        DQN_agent_Det_v4 = train_agent.train(env_Det_v4, DQN_agent_Det_v4, start_epoch, epsilon_by_epsiode, 'DQN_agent_Det_v4', episodes)
        DQN_agent_Det_v4.save_model(episodes,'DQN_agent_Det_v4')
        DQN_agent_NoFS_v4 = train_agent.train(env_NoFS_v4, DQN_agent_NoFS_v4, start_epoch, epsilon_by_epsiode, 'DQN_agent_NoFS_v4', episodes)
        DQN_agent_NoFS_v4.save_model(episodes, 'DQN_agent_NoFS_v4')
        DQN_agent_v5 = train_agent.train(env_v5, DQN_agent_v5, start_epoch, epsilon_by_epsiode, 'DQN_agent_v5', episodes)
        DQN_agent_v5.save_model(episodes, 'DQN_agent_v5')

        DDQN_agent_v4 = train_agent.train(env_v4, DDQN_agent_v4, start_epoch, epsilon_by_epsiode, 'DDQN_agent_v4', episodes)
        DDQN_agent_v4.save_model(episodes, 'DDQN_agent_v4')
        DDQN_agent_Det_v4 = train_agent.train(env_Det_v4, DDQN_agent_Det_v4, start_epoch, epsilon_by_epsiode, 'DDQN_agent_Det_v4', episodes)
        DDQN_agent_Det_v4.save_model(episodes,'DDQN_agent_Det_v4')
        DDQN_agent_NoFS_v4 = train_agent.train(env_NoFS_v4, DDQN_agent_NoFS_v4, start_epoch, epsilon_by_epsiode, 'DDQN_agent_NoFS_v4', episodes)
        DDQN_agent_NoFS_v4.save_model(episodes, 'DDQN_agent_NoFS_v4')
        DDQN_agent_v5 = train_agent.train(env_v5, DDQN_agent_v5, start_epoch, epsilon_by_epsiode, 'DDQN_agent_v5', episodes)
        DDQN_agent_v5.save_model(episodes, 'DDQN_agent_v5')
    
    ###############################################################################################################################################################
    
    else:
        if (agent == 'DQNv4'):
            DQN_agent_v4 = train_agent.train(env_v4, DQN_agent_v4, start_epoch, epsilon_by_epsiode, 'DQN_agent_v4', episodes)
            DQN_agent_v4.save_model(episodes, 'DQN_agent_v4')
        elif (agent == 'DQNv4Det'):
            DQN_agent_Det_v4 = train_agent.train(env_Det_v4, DQN_agent_Det_v4, start_epoch, epsilon_by_epsiode, 'DQN_agent_Det_v4', episodes)
            DQN_agent_Det_v4.save_model(episodes,'DQN_agent_Det_v4')
        elif (agent == 'DQNv4NoFS'):
            DQN_agent_NoFS_v4 = train_agent.train(env_NoFS_v4, DQN_agent_NoFS_v4, start_epoch, epsilon_by_epsiode, 'DQN_agent_NoFS_v4', episodes)
            DQN_agent_NoFS_v4.save_model(episodes, 'DQN_agent_NoFS_v4')
        elif (agent == 'DQNv5'):
            DQN_agent_v5 = train_agent.train(env_v5, DQN_agent_v5, start_epoch, epsilon_by_epsiode, 'DQN_agent_v5', episodes)
            DQN_agent_v5.save_model(episodes, 'DQN_agent_v5')
        elif (agent == 'DDQNv4'):
            DDQN_agent_v4 = train_agent.train(env_v4, DDQN_agent_v4, start_epoch, epsilon_by_epsiode, 'DDQN_agent_v4', episodes)
            DDQN_agent_v4.save_model(episodes, 'DDQN_agent_v4')
        elif (agent == 'DDQNv4Det'):
            DDQN_agent_Det_v4 = train_agent.train(env_Det_v4, DDQN_agent_Det_v4, start_epoch, epsilon_by_epsiode, 'DDQN_agent_Det_v4', episodes)
            DDQN_agent_Det_v4.save_model(episodes,'DDQN_agent_Det_v4')
        elif (agent == 'DDQNv4NoFS'):
            DDQN_agent_NoFS_v4 = train_agent.train(env_NoFS_v4, DDQN_agent_NoFS_v4, start_epoch, epsilon_by_epsiode, 'DDQN_agent_NoFS_v4', episodes)
            DDQN_agent_NoFS_v4.save_model(episodes, 'DDQN_agent_NoFS_v4')
        elif (agent == 'DDQNv5'):
            DDQN_agent_v5 = train_agent.train(env_v5, DDQN_agent_v5, start_epoch, epsilon_by_epsiode, 'DDQN_agent_v5', episodes)
            DDQN_agent_v5.save_model(episodes, 'DDQN_agent_v5')

###############################################################################################################################################################

if (test):
    if (agent == 'all'):
        env_v4 = train_agent.make_env('SpaceInvaders-v4', True)
        DQN_agent_v4 = DQNAgent(INPUT_SHAPE, env_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
        DQN_agent_v4.load_model(episodes, 'DQN_agent_v4')
        play(env_v4, DQN_agent_v4, 5, 'env_v4', 'DQN_agent_v4')

        env_Det_v4 = train_agent.make_env('SpaceInvadersDeterministic-v4', True)
        DQN_agent_Det_v4 = DQNAgent(INPUT_SHAPE, env_Det_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
        DQN_agent_Det_v4.load_model(episodes, 'DQN_agent_Det_v4')
        play(env_Det_v4, DQN_agent_Det_v4, 5, 'env_Det_v4', 'DQN_agent_Det_v4')

        env_NoFS_v4 = train_agent.make_env('SpaceInvadersNoFrameskip-v4', True)
        DQN_agent_NoFS_v4 = DQNAgent(INPUT_SHAPE, env_NoFS_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
        DQN_agent_NoFS_v4.load_model(episodes, 'DQN_agent_NoFS_v4')
        play(env_NoFS_v4, DQN_agent_NoFS_v4, 5, 'env_NoFS_v4', 'DQN_agent_NoFS_v4')

        env_v5 = train_agent.make_env('ALE/SpaceInvaders-v5', True)
        DQN_agent_v5 = DQNAgent(INPUT_SHAPE, env_v5.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
        DQN_agent_v5.load_model(episodes, 'DQN_agent_v5')
        play(env_v5, DQN_agent_v5, 5, 'env_v5', 'DQN_agent_v5')
        
        DDQN_agent_v4 = agent = DQNAgent(INPUT_SHAPE, env_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)
        DDQN_agent_v4.load_model(episodes, 'DDQN_agent_v4')
        play(env_v4, DDQN_agent_v4, 5, 'env_v4', 'DDQN_agent_v4')

        DDQN_agent_Det_v4 = DQNAgent(INPUT_SHAPE, env_Det_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)
        DDQN_agent_Det_v4.load_model(episodes, 'DDQN_agent_Det_v4')
        play(env_Det_v4, DDQN_agent_Det_v4, 5, 'env_Det_v4', 'DDQN_agent_Det_v4')

        DDQN_agent_NoFS_v4 = DQNAgent(INPUT_SHAPE, env_NoFS_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)
        DDQN_agent_NoFS_v4.load_model(episodes, 'DDQN_agent_NoFS_v4')
        play(env_NoFS_v4, DDQN_agent_NoFS_v4, 5, 'env_NoFS_v4', 'DDQN_agent_NoFS_v4')

        DDQN_agent_v5 = DQNAgent(INPUT_SHAPE, env_v5.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)
        DDQN_agent_v5.load_model(episodes, 'DDQN_agent_v5')
        play(env_v5, DDQN_agent_v5, 5, 'env_v5', 'DDQN_agent_v5')

    ################################################################################################################################################################

    else:
        if (agent == 'DQNv4'):
            env_v4 = train_agent.make_env('SpaceInvaders-v4', True)
            DQN_agent_v4 = DQNAgent(INPUT_SHAPE, env_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            DQN_agent_v4.load_model(episodes, 'DQN_agent_v4')
            play(env_v4, DQN_agent_v4, 5, 'env_v4', 'DQN_agent_v4')
        elif (agent == 'DQNv4Det'):
            env_Det_v4 = train_agent.make_env('SpaceInvadersDeterministic-v4', True)
            DQN_agent_Det_v4 = DQNAgent(INPUT_SHAPE, env_Det_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            DQN_agent_Det_v4.load_model(episodes, 'DQN_agent_Det_v4')
            play(env_Det_v4, DQN_agent_Det_v4, 5, 'env_Det_v4', 'DQN_agent_Det_v4')
        elif (agent == 'DQNv4NoFS'):
            env_NoFS_v4 = train_agent.make_env('SpaceInvadersNoFrameskip-v4', True)
            DQN_agent_NoFS_v4 = DQNAgent(INPUT_SHAPE, env_NoFS_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            DQN_agent_NoFS_v4.load_model(episodes, 'DQN_agent_NoFS_v4')
            play(env_NoFS_v4, DQN_agent_NoFS_v4, 5, 'env_NoFS_v4', 'DQN_agent_NoFS_v4')
        elif (agent == 'DQNv5'):
            env_v5 = train_agent.make_env('ALE/SpaceInvaders-v5', True)
            DQN_agent_v5 = DQNAgent(INPUT_SHAPE, env_v5.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
            DQN_agent_v5.load_model(episodes, 'DQN_agent_v5')
            play(env_v5, DQN_agent_v5, 5, 'env_v5', 'DQN_agent_v5')
        elif (agent == 'DDQNv4'):
            env_v4 = train_agent.make_env('SpaceInvaders-v4', True)
            DDQN_agent_v4 = agent = DQNAgent(INPUT_SHAPE, env_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)
            DDQN_agent_v4.load_model(episodes, 'DDQN_agent_v4')
            play(env_v4, DDQN_agent_v4, 5, 'env_v4', 'DDQN_agent_v4')
        elif (agent == 'DDQNv4Det'):
            env_Det_v4 = train_agent.make_env('SpaceInvadersDeterministic-v4', True)
            DDQN_agent_Det_v4 = DQNAgent(INPUT_SHAPE, env_Det_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)
            DDQN_agent_Det_v4.load_model(episodes, 'DDQN_agent_Det_v4')
            play(env_Det_v4, DDQN_agent_Det_v4, 5, 'env_Det_v4', 'DDQN_agent_Det_v4')
        elif (agent == 'DDQNv4NoFS'):
            env_NoFS_v4 = train_agent.make_env('SpaceInvadersNoFrameskip-v4', True)
            DDQN_agent_NoFS_v4 = DQNAgent(INPUT_SHAPE, env_NoFS_v4.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)
            DDQN_agent_NoFS_v4.load_model(episodes, 'DDQN_agent_NoFS_v4')
            play(env_NoFS_v4, DDQN_agent_NoFS_v4, 5, 'env_NoFS_v4', 'DDQN_agent_NoFS_v4')
        elif (agent == 'DDQNv5'):
            env_v5 = train_agent.make_env('ALE/SpaceInvaders-v5', True)
            DDQN_agent_v5 = DQNAgent(INPUT_SHAPE, env_v5.action_space.n, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)
            DDQN_agent_v5.load_model(episodes, 'DDQN_agent_v5')
            play(env_v5, DDQN_agent_v5, 5, 'env_v5', 'DDQN_agent_v5')

################################################################################################################################################################

