import traceback
import sys
from copy import deepcopy
import numpy as np
import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

from kerassurgeon import identify

ENV_NAME = 'CartPole-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape, name = 'Duda-flatten'))
model.add(Dense(nb_actions, name = 'Duda-dense'))
model.add(Activation('softmax', name = 'Duda-relu'))

print(model.summary())

memory = EpisodeParameterMemory(limit=1000, window_length=1)

cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
               batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
cem.compile()
print('\033[93m' + "Model is compiled"+'\033[0m')
cem.fit(env, nb_steps=1000, visualize=True, verbose=2)
print('\033[93m' + "Training"+'\033[0m')

cem.save_weights('cem_{}_params.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
cem.test(env, nb_episodes=5, visualize=True)

#observations = np.array((4,10))
observations = []
for _ in range(10):
    observations.append(deepcopy(env.reset()))
observations = np.asarray(observations)

for layer in cem.model.layers:
    fake_x_test = np.ndarray(shape=(10,1), dtype=float, order='F')
    # apoz = identify.get_apoz(model, layer, x_test)
    #apoz = identify.get_apoz(model, layer, observations)
    apoz = identify.get_apoz(model, layer, env.reset())
    high_apoz_channels = identify.high_apoz(apoz)
    cem.model = delete_channels(cem.model, layer, high_apoz_channels)

    print('layer name: ', layer.name)

    cem.compile(optimizer=sgd,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    reward = cem.test(env, nb_episodes=5, visualize=True)
    print('model loss after pruning: ', reward, '\n')

    results = cem.fit(env, nb_steps=100000, visualize=True, verbose=2)

    loss = cem.test(env, nb_episodes=5, visualize=True)
    print('\033[93m','model loss after retraining: ', loss, '\033[0m','\n')