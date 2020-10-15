# CartPole and Deep Reinforcement Learning

This is a Deep Neural Network designed to solve the cartpole problem, where a controlled cart must balance a pole without going too far left or right. Openai gym provides the 
cartpole environment. The cartpole environment will also provide the state/reward/done values. The model is trained by running simulations where the action to be taken is chosen by the model to either be the action with the highest possible q-value for the current state or a random action (epsilon-greedy). The model is then updated with the reward. By default, we will run 500 "games". The success of each game is measured by the number of actions the model was able to take before the game was over (AKA how long the model was able to survive). The environment considers a game to be over when either the pole has titled more than 15 degrees, the cart has moved too far left or right, or if 500 actions were taken by the model. If the
model is able to take 500 actions, we consider that game to be a success and we will stop training the model once it reaches 500 three times in a row.

# Installation

Use pip to install the necessary libraries.

```bash
pip install tensorflow, gym, numpy
```


# Usage

Run the CartPoleTraining.py file to train a model from scratch. The model will automatically bed saved once it is able to solve the game three times in a row. The amount of time
it takes to train will depend on the strength of your GPU (if you are using one). Once the model is finished training, it will prompt you for a name to save the model as. 

Run the CartPolePlaying.py file to play with a pre-trained model. A pre-trained model is included as CartPole.h5
