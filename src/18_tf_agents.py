import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

from tf_agents.environments import suite_gym
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from collections import deque
from icecream import ic


def main():
    seed = 42
    env = suite_gym.load("Breakout-v4")
    env.seed(seed)
    env.reset()

    img = env.render(mode="rgb_array")

    plt.figure(figsize=(6, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("./images/breakout_plot.png", format="png", dpi=300)
    plt.show()

    from tf_agents.environments.wrappers import ActionRepeat

    repeating_env = ActionRepeat(env, times=4)

    import tf_agents.environments.wrappers

    for name in dir(tf_agents.environments.wrappers):
        obj = getattr(tf_agents.environments.wrappers, name)
        if hasattr(obj, "__base__") and issubclass(
            obj, tf_agents.environments.wrappers.PyEnvironmentBaseWrapper
        ):
            print("{:27s} {}".format(name, obj.__doc__.split("\n")[0]))

    from functools import partial
    from gym.wrappers import TimeLimit

    limited_repeating_env = suite_gym.load(
        "Breakout-v4",
        gym_env_wrappers=[partial(TimeLimit, max_episode_steps=10000)],
        env_wrappers=[partial(ActionRepeat, times=4)],
    )

    from tf_agents.environments import suite_atari
    from tf_agents.environments.atari_preprocessing import AtariPreprocessing
    from tf_agents.environments.atari_wrappers import FrameStack4

    max_episode_steps = 27000  # <=> 108k ALE frames since 1 step = 4 frames
    environment_name = "BreakoutNoFrameskip-v4"

    env = suite_atari.load(
        environment_name,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=[AtariPreprocessing, FrameStack4],
    )

    env.seed(42)
    env.reset()
    time_step = env.step(np.array(1))  # FIRE
    for _ in range(4):
        time_step = env.step(np.array(3))  # LEFT

    def plot_observation(obs):
        # Since there are only 3 color channels, you cannot display 4 frames
        # with one primary color per frame. So this code computes the delta between
        # the current frame and the mean of the other frames, and it adds this delta
        # to the red and blue channels to get a pink color for the current frame.
        obs = obs.astype(np.float32)
        img_ = obs[..., :3]
        current_frame_delta = np.maximum(obs[..., 3] - obs[..., :3].mean(axis=-1), 0.0)
        img_[..., 0] += current_frame_delta
        img_[..., 2] += current_frame_delta
        img_ = np.clip(img_ / 150, 0, 1)
        plt.imshow(img_)
        plt.axis("off")

    plt.figure(figsize=(6, 6))
    plot_observation(time_step.observation)
    plt.tight_layout()
    plt.savefig("./images/preprocessed_breakout_plot.png", format="png", dpi=300)
    plt.show()

    from tf_agents.environments.tf_py_environment import TFPyEnvironment

    tf_env = TFPyEnvironment(env)

    from tf_agents.networks.q_network import QNetwork

    preprocessing_layer = keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.0)
    conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    fc_layer_params = [512]

    q_net = QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
    )

    from tf_agents.agents.dqn.dqn_agent import DqnAgent

    # see TF-agents issue #113
    # optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0,
    #                                     epsilon=0.00001, centered=True)

    train_step = tf.Variable(0)
    update_period = 4  # run a training step every 4 collect steps
    optimizer = tf.compat.v1.train.RMSPropOptimizer(
        learning_rate=2.5e-4, decay=0.95, momentum=0.0, epsilon=0.00001, centered=True
    )
    epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0,  # initial ε
        decay_steps=250000 // update_period,  # <=> 1,000,000 ALE frames
        end_learning_rate=0.01,
    )  # final ε
    agent = DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=2000,  # <=> 32,000 ALE frames
        td_errors_loss_fn=keras.losses.Huber(reduction="none"),
        gamma=0.99,  # discount factor
        train_step_counter=train_step,
        epsilon_greedy=lambda: epsilon_fn(train_step),
    )
    agent.initialize()

    from tf_agents.replay_buffers import tf_uniform_replay_buffer

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec, batch_size=tf_env.batch_size, max_length=1000000
    )

    replay_buffer_observer = replay_buffer.add_batch

    class ShowProgress:
        def __init__(self, total):
            self.counter = 0
            self.total = total

        def __call__(self, trajectory):
            if not trajectory.is_boundary():
                self.counter += 1
            if self.counter % 100 == 0:
                print("\r{}/{}".format(self.counter, self.total), end="")

    from tf_agents.metrics import tf_metrics

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    from tf_agents.eval.metric_utils import log_metrics
    import logging

    logging.getLogger().setLevel(logging.INFO)
    log_metrics(train_metrics)

    from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

    collect_driver = DynamicStepDriver(
        tf_env,
        agent.collect_policy,
        observers=[replay_buffer_observer] + train_metrics,
        num_steps=update_period,
    )  # collect 4 steps for each training iteration

    from tf_agents.policies.random_tf_policy import RandomTFPolicy

    initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
    init_driver = DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch, ShowProgress(20000)],
        num_steps=20000,
    )  # <=> 80,000 ALE frames
    final_time_step, final_policy_state = init_driver.run()


if __name__ == "__main__":
    main()
