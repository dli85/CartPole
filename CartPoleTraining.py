import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf

EPSILON = 1.0
EPS_DECAY_EACH_TIME = 0.0005
START_DECAY_AT_EPISODE = 50
EPS_MIN = 0.001
DISCOUNT = 0.95


class Agent:
    def __init__(self, obs_space_size, actions_size):
        self.epsilon = EPSILON
        self.eps_decay = EPS_DECAY_EACH_TIME
        self.start_dec = START_DECAY_AT_EPISODE
        self.epsilon_min = EPS_MIN
        self.discount = DISCOUNT

        self.os_size = obs_space_size
        self.actions_size = actions_size
        self.memory = deque(maxlen=10000)

        #Create the model
        inputs = tf.keras.Input(shape=(self.os_size,))
        x = tf.keras.layers.Dense(512, activation='relu')(inputs)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.actions_size, activation='linear')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mse", optimizer='adam', metrics=["accuracy"])
        self.model = model

    def save_model(self, name):
        self.model.save(name)
        print("Model has been saved as", name)

    def load_prev_saved_model(self, name):
        try:
            self.model = tf.keras.models.load_model(name)
        except:
            print("Failed to load model. Maybe you typed the name wrong?")
            return False
        else:
            print("Model sucessfully loaded!")
            return True


    def exp_replay(self):
        if len(self.memory) < 64:
            return

        batch = random.sample(self.memory, 64)

        current_state = np.zeros((64, self.os_size))
        next_state = np.zeros((64, self.os_size))

        for i in range(64):
            current_state[i] = batch[i][0]
            next_state[i] = batch[i][3]

        target = self.model.predict(current_state)
        target_next = self.model.predict(next_state)

        for i in range(64):
            if batch[i][4]:
                target[i][batch[i][1]] = batch[i][2]
            else:
                target[i][batch[i][1]] = batch[i][2] + self.discount * (np.amax(target_next[i]))

        self.model.fit(current_state, target)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.actions_size)
            return action
        else:
            action = np.argmax(self.model.predict(state))
            return action

    def add_to_mem(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def start_training(self, user_input_episodes, env):
        if user_input_episodes:
            num_episodes = int(user_input_episodes.strip())
        else:
            num_episodes = 500

        doRender = False

        maxInARow = 0

        for episode in range(num_episodes):
            current_state = env.reset()
            done = False
            action_num = 0
            current_state = np.reshape(current_state, [1, self.os_size])

            while not done:

                if(doRender):
                    env.render()

                action_num += 1
                action = self.select_action(current_state)
                next_state, reward, done, _ = env.step(action)
                if not done:
                    pass
                else:
                    reward = -100

                next_state = np.reshape(next_state, [1, self.os_size])
                self.add_to_mem(current_state, action, reward, next_state, done)
                current_state = next_state

                if done:
                    print("Episode:", episode, "Actions taken:", action_num, "Epsilon:", self.epsilon)
                    break

                self.exp_replay()
                if(episode >= self.start_dec and self.epsilon >= self.epsilon_min):
                    self.epsilon -= self.eps_decay

            if action_num == 500:
                maxInARow+=1
                if(maxInARow == 3):
                    mod_name = input("Your model has finished training! What name would you like to save it as(Your name should end with .h5): ")
                    self.save_model(mod_name)
                    break
            else:
                maxInARow = 0

if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    agent = Agent(len(env.observation_space.high), env.action_space.n)

    user_input_episodes = input("How many episodes would you like to train on(Default is 500): ")
    agent.start_training(user_input_episodes, env)

