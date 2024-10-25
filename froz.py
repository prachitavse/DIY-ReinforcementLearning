import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import deque

# Initialize the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n

# Function to visualize the environment grid in a single plot
def visualize_environment(env, state, episode, step, fig, ax):
    grid_size = (4, 4)
    grid = np.zeros(grid_size)

    # Mark the goal
    goal_state = 15
    grid[goal_state // 4, goal_state % 4] = 1  # Mark the goal with a 1

    # Mark the holes
    holes = [5, 7, 11, 12]
    for hole in holes:
        grid[hole // 4, hole % 4] = 0.5  # Mark holes with 0.5

    # Mark the agent's position
    agent_position = state
    grid[agent_position // 4, agent_position % 4] = 0.75  # Mark agent with 0.75

    ax.clear()  # Clear the previous plot
    cmap = colors.ListedColormap(['lightblue', 'yellow', 'blue', 'red'])
    bounds = [0, 0.25, 0.5, 0.75, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_xticks([])  # Remove x ticks
    ax.set_yticks([])  # Remove y ticks
    ax.set_title(f'Episode {episode} / Step {step}')
    plt.draw()
    plt.pause(0.05)  # Short pause to allow for real-time updates

# Build the actor model
actor_model = tf.keras.Sequential([
    layers.Dense(24, input_dim=state_size, activation='relu'),
    layers.Dense(24, activation='relu'),
    layers.Dense(action_size, activation='softmax')
])

# Build the critic model
critic_model = tf.keras.Sequential([
    layers.Dense(24, input_dim=state_size, activation='relu'),
    layers.Dense(24, activation='relu'),
    layers.Dense(1)  # Single output for the state value
])

# Optimizers for the actor and critic
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Training parameters
gamma = 0.99
num_episodes = 1000
max_steps = 100
success_threshold = 0.9  # Early stopping success rate threshold

# Tracking success rate over recent episodes
recent_rewards = deque(maxlen=100)  # Store rewards for the last 100 episodes

# Set up the plotting window outside the loop
fig, ax = plt.subplots()

# Main training loop
for episode in range(num_episodes):
    state = env.reset()[0]  # Reset environment and get initial state
    episode_reward = 0
    done = False

    with tf.GradientTape(persistent=True) as tape:  # Persistent to allow reuse for both models
        for step in range(max_steps):
            # Visualize environment at each step
            visualize_environment(env, state, episode, step, fig, ax)

            state_one_hot = np.identity(state_size)[state]  # One-hot encode state
            state_input = np.array([state_one_hot])

            # Get action probabilities from actor and sample an action
            action_probs = actor_model(state_input, training=True)
            action = np.random.choice(action_size, p=action_probs.numpy().flatten())

            # Take action and observe result
            next_state, reward, done, _, _ = env.step(action)
            next_state_one_hot = np.identity(state_size)[next_state]
            next_state_input = np.array([next_state_one_hot])

            # Compute state values and advantage
            state_value = critic_model(state_input, training=True)[0, 0]
            next_state_value = critic_model(next_state_input, training=True)[0, 0]
            advantage = reward + gamma * next_state_value - state_value

            # Compute actor and critic losses
            actor_loss = -tf.math.log(action_probs[0, action]) * advantage
            critic_loss = tf.square(advantage)

            episode_reward += reward
            state = next_state  # Update the state

            # Stop if episode is done
            if done:
                break

    # Compute gradients and apply updates
    actor_gradients = tape.gradient(actor_loss, actor_model.trainable_variables)
    critic_gradients = tape.gradient(critic_loss, critic_model.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_gradients, actor_model.trainable_variables))
    critic_optimizer.apply_gradients(zip(critic_gradients, critic_model.trainable_variables))

    # Store success/failure (1 for reaching the goal, 0 otherwise)
    recent_rewards.append(1 if reward == 1 else 0)  # Reward 1 indicates success in FrozenLake

    # Print episode stats every 10 episodes
    if episode % 10 == 0:
        success_rate = np.mean(recent_rewards)
        print(f"Episode {episode}, Reward: {episode_reward}, Success Rate: {success_rate * 100:.2f}%")

    # Check early stopping condition
    if len(recent_rewards) == recent_rewards.maxlen and np.mean(recent_rewards) >= success_threshold:
        print(f"Early stopping at episode {episode} with success rate {np.mean(recent_rewards) * 100:.2f}%")
        break

env.close()

