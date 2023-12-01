import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent, SARSAAgent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory

import warnings
warnings.filterwarnings('ignore')


def build_model(states, actions, num_hidden_layers=4, num_nodes=24):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    for l in range(num_hidden_layers):
        model.add(Dense(num_nodes, activation='relu'))

    model.add(Dense(actions, activation='linear'))

    return model


def build_agent(model, actions, policy, limit, warmup=200, agent_name='DQNAgent', target_update=1e-1, gamma=0.99):
    if agent_name == 'DQNAgent':
        memory = SequentialMemory(limit=limit, window_length=1)
        agent = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=warmup,
                         target_model_update=target_update)

    elif agent_name == 'SARSAAgent':
        agent = SARSAAgent(model=model, policy=policy, nb_actions=actions,  nb_steps_warmup=warmup, gamma=gamma)

    return agent


def train(env, agent, limit=10000, test_episodes=20):
    agent.fit(env, nb_steps=limit, visualize=False, verbose=1)
    agent.save_weights('agent_weights.h5f', overwrite=True)

    scores = agent.test(env, nb_episodes=test_episodes, visualize=False)
    print('Mean of reward: '+str(np.mean(scores.history['episode_reward'])))


def load_evaluate(agent, episodes=5, visualize=True):
    agent.load_weights('agent_weights.h5f')
    _ = agent.test(env, nb_episodes=episodes, visualize=visualize)

# -------Hyper parameters-----------
limit = 30000
learning_rate = 1e-3
num_hidden_layers = 2
num_nodes = 25
warmup = 200
target_update = 1e-3
gamma = 0.99

agent_names = ['DQNAgent', 'SARSAAgent']
polices = [BoltzmannQPolicy(), EpsGreedyQPolicy(), GreedyQPolicy()]
police_names = ['BoltzmannQPolicy', 'EpsGreedyQPolicy', 'GreedyQPolicy']


# -------Training------------
env = gym.make("LunarLander-v2")
states = env.observation_space.shape[0]
actions = env.action_space.n

print('layers: '+str(num_hidden_layers)+', nodes: '+str(num_nodes))
model = build_model(states, actions, num_hidden_layers=num_hidden_layers, num_nodes=num_nodes)

for an in agent_names:
    for police, name in zip(polices, police_names):
        print('agent: '+str(an)+', police: '+str(name)+', warmup: '+str(warmup)+', learning rate: '+str(learning_rate)
              +', target update: '+str(target_update)+', gamma: '+str(gamma))
        agent = build_agent(model=model, actions=actions, policy=police, limit=limit, warmup=warmup,
                            agent_name=an, target_update=target_update, gamma=gamma)
        agent.compile(Adam(lr=learning_rate), metrics=['mae'])

        train(env=env, agent=agent, limit=limit)

