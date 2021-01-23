import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import gym

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from collections import deque
from icecream import ic


def basic_policy(seed=42):
    env = gym.make("CartPole-v1")

    def policy(obs_):
        angle = obs_[2]
        return 0 if angle < 0 else 1

    totals = []
    env.seed(seed)
    for episode in range(200):
        episode_rewards = 0
        obs = env.reset()
        for step in range(200):
            action = policy(obs)
            # action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                print(f"episode done with step {step}")
                break
        totals.append(episode_rewards)

    print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))


def render_policy_net(model, n_max_steps=500, seed=42):
    env = gym.make("CartPole-v1")
    env.seed(seed)
    obs = env.reset()

    frames = []
    for step in range(n_max_steps):
        frames.append(env.render(mode="rgb_array"))
        left_proba = model.predict(obs.reshape(1, -1))
        action = int(np.random.rand() > left_proba)
        obs, reward, done, info = env.step(action)
        if done:
            print(f"episode done with step {step}")
            break
    env.close()

    return frames


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return (patch,)


def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis("off")
    anim = animation.FuncAnimation(
        fig,
        update_scene,
        fargs=(frames, patch),
        frames=len(frames),
        repeat=repeat,
        interval=interval,
    )
    plt.close()
    return anim


def render_policy(seed=42):
    env = gym.make("CartPole-v1")

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    n_inputs = env.observation_space.shape[0]

    model = models.Sequential()
    model.add(layers.Dense(5, activation="elu", input_shape=(n_inputs,)))
    model.add(layers.Dense(1, activation="sigmoid"))

    n_environments = 50
    n_iterations = 5000

    envs = [gym.make("CartPole-v1") for _ in range(n_environments)]

    for index, env in enumerate(envs):
        env.seed(index)
    observations = [env.reset() for env in envs]
    optimizer = optimizers.RMSprop()
    loss_fn = tf.keras.losses.binary_crossentropy

    for iteration in range(n_iterations):
        # if angle < 0, we want proba(left) = 1., or else proba(left) = 0.
        target_probas = np.array([([1.0] if obs[2] < 0 else [0.0]) for obs in observations])
        with tf.GradientTape() as tape:
            left_probas = model(np.array(observations))
            loss = tf.reduce_mean(loss_fn(target_probas, left_probas))
        print("\rIteration: {}, Loss: {:.3f}".format(iteration, loss.numpy()), end="")
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        actions = (np.random.rand(n_environments, 1) > left_probas.numpy()).astype(np.int32)
        for env_index, env in enumerate(envs):
            obs, reward, done, info = env.step(actions[env_index][0])
            observations[env_index] = obs if not done else env.reset()

    for env in envs:
        env.close()

    frames = render_policy_net(model)
    plot_animation(frames)


def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = tf.random.uniform([1, 1]) > left_proba
        y_target = tf.constant([[1.0]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))
    return obs, reward, done, grads


def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads


def discount_rewards(rewards, discount_rate):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_rate
    return discounted


def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [
        (discounted_rewards - reward_mean) / reward_std
        for discounted_rewards in all_discounted_rewards
    ]


def policy_gradients(seed=42):
    n_iterations = 150
    n_episodes_per_update = 10
    n_max_steps = 200
    discount_rate = 0.95

    optimizer = optimizers.Adam(lr=0.01)
    loss_fn = tf.keras.losses.binary_crossentropy

    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = models.Sequential(
        [
            layers.Dense(5, activation="elu", input_shape=(4,)),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    env = gym.make("CartPole-v1")
    env.seed(seed)

    for iteration in range(n_iterations):
        all_rewards, all_grads = play_multiple_episodes(
            env, n_episodes_per_update, n_max_steps, model, loss_fn
        )
        total_rewards = sum(map(sum, all_rewards))
        mean_rewards = total_rewards / n_episodes_per_update
        print(f"Iteration: {iteration}, mean reward: {mean_rewards:.1f}")
        all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        all_mean_grads = []
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [
                    final_reward * all_grads[episode_index][step][var_index]
                    for episode_index, final_rewards in enumerate(all_final_rewards)
                    for step, final_reward in enumerate(final_rewards)
                ],
                axis=0,
            )
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

    env.close()

    frames = render_policy_net(model)
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=30, metadata=dict(artist="mecha2k"), bitrate=1800)
    anim = plot_animation(frames)
    anim.save("./images/policygrad.mp4", writer=writer)


def markov_chain(seed=42):
    np.random.seed(seed)
    transition_probabilities = [  # shape=[s, s']
        [0.7, 0.2, 0.0, 0.1],  # from s0 to s0, s1, s2, s3
        [0.0, 0.0, 0.9, 0.1],  # from s1 to ...
        [0.0, 1.0, 0.0, 0.0],  # from s2 to ...
        [0.0, 0.0, 0.0, 1.0],  # from s3 to ...
    ]
    n_max_steps = 50

    def print_sequence():
        current_state = 0
        print("States:", end=" ")
        for step in range(n_max_steps):
            print(current_state, end=" ")
            if current_state == 3:
                break
            current_state = np.random.choice(range(4), p=transition_probabilities[current_state])
        else:
            print("...", end="")
        print()

    for _ in range(10):
        print_sequence()


def q_value_iter(seed=42):
    np.random.seed(seed)
    # fmt: off
    transition_probabilities = [  # shape=[s, a, s']
        [[0.7, 0.3, 0.0],
         [1.0, 0.0, 0.0],
         [0.8, 0.2, 0.0]],
        [[0.0, 1.0, 0.0],
         None,
         [0.0, 0.0, 1.0]],
        [None,
         [0.8, 0.1, 0.1],
         None],
    ]
    rewards = [  # shape=[s, a, s']
        [[+10, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, -50]],
        [[0, 0, 0],
         [+40, 0, 0],
         [0, 0, 0]],
    ]
    possible_actions = [[0, 1, 2], [0, 2], [1]]
    # fmt: on

    Q_values = np.full((3, 3), -np.inf)  # -np.inf for impossible actions
    for state, actions in enumerate(possible_actions):
        Q_values[state, actions] = 0.0  # for all possible actions

    gamma = 0.90  # the discount factor
    history1 = []
    for iteration in range(50):
        Q_prev = Q_values.copy()
        history1.append(Q_prev)
        for s in range(3):
            for a in possible_actions[s]:
                Q_values[s, a] = np.sum(
                    [
                        transition_probabilities[s][a][sp]
                        * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
                        for sp in range(3)
                    ]
                )
    # history1 = np.array(history1)
    ic(Q_values)
    ic(np.argmax(Q_values, axis=1))

    Q_values = np.full((3, 3), -np.inf)  # -np.inf for impossible actions
    for state, actions in enumerate(possible_actions):
        Q_values[state, actions] = 0.0  # for all possible actions

    gamma = 0.95  # the discount factor
    for iteration in range(50):
        Q_prev = Q_values.copy()
        for s in range(3):
            for a in possible_actions[s]:
                Q_values[s, a] = np.sum(
                    [
                        transition_probabilities[s][a][sp]
                        * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
                        for sp in range(3)
                    ]
                )
    ic(Q_values)
    ic(np.argmax(Q_values, axis=1))


def deep_q_network(seed=42):
    env = gym.make("CartPole-v1")
    input_shape = [4]  # == env.observation_space.shape
    n_outputs = 2  # == env.action_space.n

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    env.seed(42)

    model = models.Sequential()
    model.add(layers.Dense(32, activation="elu", input_shape=input_shape))
    model.add(layers.Dense(32, activation="elu"))
    model.add(layers.Dense(n_outputs))

    replay_memory = deque(maxlen=2000)

    def epsilon_greedy_policy(state_, epsilon_=0):
        if np.random.rand() < epsilon_:
            return np.random.randint(2)
        else:
            Q_values = model.predict(state_[np.newaxis])
            return np.argmax(Q_values[0])

    def sample_experiences(batch_size_):
        indices = np.random.randint(len(replay_memory), size=batch_size_)
        batch = [replay_memory[index] for index in indices]
        states, actions, rewards_, next_states, dones = [
            np.array([experience[field_index] for experience in batch]) for field_index in range(5)
        ]
        return states, actions, rewards_, next_states, dones

    def play_one_step_q(env_, state_, epsilon_):
        action_ = epsilon_greedy_policy(state_, epsilon_)
        next_state, reward_, done_, info_ = env_.step(action_)
        replay_memory.append((state_, action_, reward_, next_state, done_))
        return next_state, reward_, done_, info_

    batch_size = 32
    discount_rate = 0.95
    optimizer = optimizers.Adam(lr=1e-3)
    loss_fn = tf.keras.losses.mean_squared_error

    def training_step(batch_size_):
        experiences = sample_experiences(batch_size_)
        states, actions, rewards_, next_states, dones = experiences
        next_Q_values = model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = rewards_ + (1 - dones) * discount_rate * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    rewards = []
    best_score = 0
    best_weights = 0
    for episode in range(10):
        obs = env.reset()
        step, epsilon = 0, 0
        for step in range(200):
            epsilon = max(1 - episode / 500, 0.01)
            obs, reward, done, info = play_one_step_q(env, obs, epsilon)
            if done:
                break
        rewards.append(step)
        if step > best_score:
            best_weights = model.get_weights()
            best_score = step
        if episode % 50 == 0:
            print(f"Episode: {episode:3d}, Steps: {step+1:3d}, eps: {epsilon:.3f}")
        if episode > 50:
            training_step(batch_size)

    model.set_weights(best_weights)

    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Sum of rewards", fontsize=14)
    plt.tight_layout()
    plt.savefig("./images/dqn_rewards_plot.png", format="png", dpi=300)
    plt.show()

    frames = []
    env.seed(seed)
    state = env.reset()
    for step in range(200):
        action = epsilon_greedy_policy(state)
        state, reward, done, info = env.step(action)
        if done:
            break
        img = env.render(mode="rgb_array")
        frames.append(img)
    env.close()

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=30, metadata=dict(artist="mecha2k"), bitrate=1800)
    anim = plot_animation(frames)
    anim.save("./images/deep_q_network.mp4", writer=writer)


def main():
    seed = 100
    # basic_policy(seed)
    # render_policy(seed)
    # policy_gradients(seed)
    # markov_chain(seed)
    # q_value_iter(seed)
    deep_q_network(seed)


if __name__ == "__main__":
    main()
