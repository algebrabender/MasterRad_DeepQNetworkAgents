import time
from Utils.preporcessing import preprocess_frame, stack_frame

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (8, -12, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

def test(env, agent):
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

    return score

def play(env, agent, n_of_times, env_name, agent_name):
    with open('results_{}_{}.txt'.format(env_name, agent_name), 'w') as f:
        
        for i in range (0, n_of_times):
            start_time = time.time()
            score = test(env, agent)
            end_time = time.time()
            duration = end_time - start_time
            f.write('Env: {} | Agent: {} | Iteration: {} | Iteration Duration: {}s | Score: {}\n'.format(env_name, agent_name, i+1, round(duration, 2), score))
            print('Env: {} | Agent: {} | Iteration: {} | Iteration Duration: {}s | Score: {}'.format(env_name, agent_name, i+1, round(duration, 2), score))