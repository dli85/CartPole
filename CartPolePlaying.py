import tensorflow as tf
import gym
import numpy as np

if __name__ == '__main__':
    name = input("What is the name of your model(Should end with .h5): ")
    model = tf.keras.models.load_model(name)
    trials = int(input("How many games would you like to run: "))

    env = gym.make('CartPole-v1')

    for trial in range(trials):
        action_num = 0
        current_state = env.reset()
        current_state = np.reshape(current_state, [1, len(env.observation_space.high)])

        done = False
        while not done:

            action = np.argmax(model.predict(current_state))
            next_state, reward, done, _ = env.step(action)
            current_state = np.reshape(next_state, [1, len(env.observation_space.high)])
            action_num += 1

            if done:
                print("We were able to take", action_num, "actions")
                break

            env.render()
