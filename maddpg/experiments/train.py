import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import random
import analyze
import os
import rewards
import maddpg.common.tf_util as U
from maddpg.trainer.rmaddpg import _RMADDPGAgentTrainer
import utils
from network import mlp_model, lstm_fc_model

os.environ["CUDA_VISIBLE_DEVICES"]="3"

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Output Filename
    parser.add_argument("--commit_num", type=str, default="0", help="commit number?")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    parser.add_argument("--seed", type=int, default=1, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--update-freq", type=int, default=100, help="number of timesteps trainer should be updated ")
    parser.add_argument("--no-comm", action="store_true", default=False) # for analysis purposes
    parser.add_argument("--critic-lstm", action="store_true", default=False)
    parser.add_argument("--actor-lstm", action="store_true", default=False)
    parser.add_argument("--centralized-actor", action="store_true", default=False)
    parser.add_argument("--with-comm-budget", action="store_true", default=False)
    parser.add_argument("--analysis", type=str, default="", help="type of analysis") # time, pos, argmax
    parser.add_argument("--commit-num", type=str, default="", help="name of the experiment")
    parser.add_argument("--sync-sampling", action="store_true", default=False)
    parser.add_argument("--tracking", action="store_true", default=False)
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./saved_policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./saved_policy/", help="directory in which training state and model are loaded")
    parser.add_argument("--test-actor-q", action="store_true", default=False)
    # Evaluation
    parser.add_argument("--graph", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--metrics-filename", type=str, default="", help="name of metrics filename")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, rewards.sim_higher_arrival_reward,  scenario.observation) # , done_callback=scenario.done)
        # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,  scenario.observation) # , done_callback=scenario.done)
    return env


def get_lstm_states(_type, trainers):
    if _type == 'p':
        return [agent.p_c for agent in trainers], [agent.p_h for agent in trainers]
    if _type == 'q':
        return [agent.q_c for agent in trainers], [agent.q_h for agent in trainers]
    else:
        raise ValueError("unknown type")

def update_critic_lstm(trainers, obs_n, action_n, p_states):
    obs_n = [o[None] for o in obs_n]
    action_n = [a[None] for a in action_n]
    q_c_n = [trainer.q_c for trainer in trainers]
    q_h_n = [trainer.q_h for trainer in trainers]
    p_c_n, p_h_n = p_states if p_states else [None, None]

    for trainer in trainers:
        q_val, (trainer.q_c, trainer.q_h) = trainer.q_debug['q_values'](*(obs_n + action_n + q_c_n + q_h_n))

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    trainer = _RMADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, mlp_model, lstm_fc_model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    # agents get id after geting adversaries
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i,  mlp_model, lstm_fc_model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def create_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

class trainTest():
    def init(self, arglist):
        self.sess = tf.InteractiveSession()
        print("qwelrjasleiufjalksdf")
        print(self.sess)

        # To make sure that training and testing are based on diff seeds
        if arglist.restore:
            create_seed(np.random.randint(2))
        else:
            create_seed(arglist.seed)

        # Create environment
        self.env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        self.obs_shape_n = [self.env.observation_space[i].shape for i in range(self.env.n)]
        self.num_adversaries = min(self.env.n, arglist.num_adversaries)
        self.trainers = get_trainers(self.env, self.num_adversaries, self.obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        self.episode_rewards = [0.0]  # sum of rewards for all agents
        self.agent_rewards = [[0.0] for _ in range(self.env.n)]  # individual agent reward
        self.final_ep_rewards = []  # sum of rewards for training curve
        self.final_ep_ag_rewards = []  # agent rewards for training curve
        self.agent_info = [[[]]]  # placeholder for benchmarking info
        self.saver = tf.train.Saver()
        self.obs_n = self.env.reset()
        self.train_step = 0
        self.t_start = time.time()
        self.new_episode = True # start of a new episode (used for replay buffer)
        self.start_saving_comm = False

        if arglist.graph:
            print("Setting up graph writer!")
            self.writer = tf.summary.FileWriter("learning_curves/graph",sess.graph)

        if arglist.analysis:
            print("Starting analysis on {}...".format(arglist.analysis))
            if arglist.analysis != 'video':
                analyze.run_analysis(arglist, self.env, self.trainers)
            return # should be a single run

    def update(self, arglist, obs_n, rew_n, done_n, info_n, terminal):
        # info_n is false only when the very first data was created
        if info_n != False:
            done = all(done_n)

            # collect experience
            for i, agent in enumerate(self.trainers):
                # do this every iteration
                if arglist.critic_lstm and arglist.actor_lstm:
                    agent.experience(self.prev_obs_n[i], self.action_n[i], rew_n[i],
                                    obs_n[i], done_n[i], # terminal,
                                    self.p_in_c_n[i][0], self.p_in_h_n[i][0],
                                    self.p_out_c_n[i][0], self.p_out_h_n[i][0],
                                    self.q_in_c_n[i][0], self.q_in_h_n[i][0],
                                    self.q_out_c_n[i][0], self.q_out_h_n[i][0], self.new_episode)
                elif arglist.critic_lstm:
                    agent.experience(self.prev_obs_n[i], self.action_n[i], rew_n[i],
                                    obs_n[i], done_n[i], # terminal,
                                    self.q_in_c_n[i][0], self.q_in_h_n[i][0],
                                    self.q_out_c_n[i][0], self.q_out_h_n[i][0],self.new_episode)
                elif arglist.actor_lstm:
                    agent.experience(self.prev_obs_n[i], self.action_n[i], rew_n[i],
                                    obs_n[i], done_n[i], # terminal,
                                    self.p_in_c_n[i][0], self.p_in_h_n[i][0],
                                    self.p_out_c_n[i][0], self.p_out_h_n[i][0],
                                    self.new_episode)
                else:
                    agent.experience(self.prev_obs_n[i], self.action_n[i], rew_n[i],
                                    obs_n[i], done_n[i], # terminal,
                                    self.new_episode)

            # Adding rewards
            if arglist.tracking:
                for i, a in enumerate(self.trainers):
                    # if arglist.num_episodes - len(self.episode_rewards) <= 1000:
                    #     a.tracker.record_information("goal", np.array(self.env.world.landmarks[0].state.p_pos))
                    #     a.tracker.record_information("position",np.array(self.env.world.agents[i].state.p_pos))
                    a.tracker.record_information("ag_reward", rew_n[i])
                    a.tracker.record_information("team_dist_reward", info_n["team_dist"][i])
                    a.tracker.record_information("team_diff_reward", info_n["team_diff"][i])

            # Closing graph writer
            if arglist.graph:
                self.writer.close()
            for i, rew in enumerate(rew_n):
                self.episode_rewards[-1] += rew
                self.agent_rewards[i][-1] += rew

            # If an episode was finished, reset internal values
            if done or terminal:
                self.new_episode = True
                # reset trainers
                if arglist.actor_lstm or arglist.critic_lstm:
                    for agent in self.trainers:
                        agent.reset_lstm()
                if arglist.tracking:
                    for agent in self.trainers:
                        agent.tracker.reset()
                self.episode_rewards.append(0)
                for a in self.agent_rewards:
                    a.append(0)
                self.agent_info.append([[]])
            else:
                self.new_episode=False

            # increment global step counter
            self.train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    self.agent_info[-1][i].append(info_n['n'])
                if self.train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(self.agent_info[:-1], fp)
                    return
            # otherwise training
            else:
                # update all trainers, if not in display or benchmark mode
                loss = None

                # get same episode sampling
                if arglist.sync_sampling:
                    inds = [random.randint(0, len(self.trainers[0].replay_buffer._storage)-1) for i in range(arglist.batch_size)]
                else:
                    inds = None

                for agent in self.trainers:
                    # if arglist.lstm:
                    #     agent.preupdate(inds=inds)
                    # else:
                    agent.preupdate(inds)
                for agent in self.trainers:
                    loss = agent.update(self.trainers, self.train_step)
                    if loss is None: continue

                # save model, display training output
                if terminal and (len(self.episode_rewards) % arglist.save_rate == 0):
                    U.save_state(arglist.save_dir, saver=self.saver)
                    # print statement depends on whether or not there are adversaries
                    if self.num_adversaries == 0:
                        print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                            self.train_step, len(self.episode_rewards), np.mean(self.episode_rewards[-arglist.save_rate:]), round(time.time()-self.t_start, 3)))
                    else:
                        print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                            self.train_step, len(self.episode_rewards), np.mean(self.episode_rewards[-arglist.save_rate:]),
                            [np.mean(rew[-arglist.save_rate:]) for rew in self.agent_rewards], round(time.time()-self.t_start, 3)))
                    self.t_start = time.time()
                    # Keep track of final episode reward
                    self.final_ep_rewards.append(np.mean(self.episode_rewards[-arglist.save_rate:]))
                    for rew in self.agent_rewards:
                        self.final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))


        if arglist.actor_lstm:
            # get critic input states
            self.p_in_c_n, self.p_in_h_n = get_lstm_states('p', self.trainers) # num_trainers x 1 x 1 x 64
        if arglist.critic_lstm:
            self.q_in_c_n, self.q_in_h_n = get_lstm_states('q', self.trainers) # num_trainers x 1 x 1 x 64

        # get action
        self.action_n = [agent.action(obs) for agent, obs in zip(self.trainers, obs_n)]
        if arglist.critic_lstm:
            # get critic output states
            p_states = [self.p_in_c_n, self.p_in_h_n] if arglist.actor_lstm else []
            update_critic_lstm(self.trainers, obs_n, self.action_n, p_states)
            self.q_out_c_n, self.q_out_h_n = get_lstm_states('q', self.trainers) # num_trainers x 1 x 1 x 64
        if arglist.actor_lstm:
            self.p_out_c_n, self.p_out_h_n = get_lstm_states('p', self.trainers) # num_trainers x 1 x 1 x 64

        self.prev_obs_n = obs_n

        return self.action_n



    def loop(self, arglist):
        print('Starting iterations...')
        while True:
            if arglist.actor_lstm:
                # get critic input states
                p_in_c_n, p_in_h_n = get_lstm_states('p', self.trainers) # num_trainers x 1 x 1 x 64
            if arglist.critic_lstm:
                q_in_c_n, q_in_h_n = get_lstm_states('q', self.trainers) # num_trainers x 1 x 1 x 64

            # get action
            action_n = [agent.action(obs) for agent, obs in zip(self.trainers,self.obs_n)]
            if arglist.critic_lstm:
                # get critic output states
                p_states = [p_in_c_n, p_in_h_n] if arglist.actor_lstm else []
                update_critic_lstm(self.trainers, self.obs_n, action_n, p_states)
                q_out_c_n, q_out_h_n = get_lstm_states('q', self.trainers) # num_trainers x 1 x 1 x 64
            if arglist.actor_lstm:
                p_out_c_n, p_out_h_n = get_lstm_states('p', self.trainers) # num_trainers x 1 x 1 x 64

            # environment step
            new_obs_n, rew_n, done_n, info_n = self.env.step(action_n)
            self.episode_step += 1
            done = all(done_n)
            terminal = (self.episode_step >= arglist.max_episode_len)

            # collect experience
            for i, agent in enumerate(self.trainers):
                num_episodes = len(self.episode_rewards)
                # do this every iteration
                if arglist.critic_lstm and arglist.actor_lstm:
                    agent.experience(self.obs_n[i], action_n[i], rew_n[i],
                                    new_obs_n[i], done_n[i], # terminal,
                                    p_in_c_n[i][0], p_in_h_n[i][0],
                                    p_out_c_n[i][0], p_out_h_n[i][0],
                                    q_in_c_n[i][0], q_in_h_n[i][0],
                                    q_out_c_n[i][0], q_out_h_n[i][0], self.new_episode)
                elif arglist.critic_lstm:
                    agent.experience(obs_n[i], action_n[i], rew_n[i],
                                    new_obs_n[i], done_n[i], # terminal,
                                    q_in_c_n[i][0], q_in_h_n[i][0],
                                    q_out_c_n[i][0], q_out_h_n[i][0],self.new_episode)
                elif arglist.actor_lstm:
                    agent.experience(obs_n[i], action_n[i], rew_n[i],
                                    new_obs_n[i], done_n[i], # terminal,
                                    p_in_c_n[i][0], p_in_h_n[i][0],
                                    p_out_c_n[i][0], p_out_h_n[i][0],
                                    self.new_episode)
                else:
                    agent.experience(obs_n[i], action_n[i], rew_n[i],
                                    new_obs_n[i], done_n[i], # terminal,
                                    self.new_episode)

                self.obs_n = new_obs_n

            # Adding rewards
            if arglist.tracking:
                for i, a in enumerate(self.trainers):
                    if arglist.num_episodes - len(self.episode_rewards) <= 1000:
                        a.tracker.record_information("goal", np.array(self.env.world.landmarks[0].state.p_pos))
                        a.tracker.record_information("position",np.array(self.env.world.agents[i].state.p_pos))
                    a.tracker.record_information("ag_reward", rew_n[i])
                    a.tracker.record_information("team_dist_reward", info_n["team_dist"][i])
                    a.tracker.record_information("team_diff_reward", info_n["team_diff"][i])

            # Closing graph writer
            if arglist.graph:
                self.writer.close()
            for i, rew in enumerate(rew_n):
                self.episode_rewards[-1] += rew
                self.agent_rewards[i][-1] += rew

            if done or terminal:
                self.new_episode = True
                num_episodes = len(self.episode_rewards)
                self.obs_n = self.env.reset()
                # reset trainers
                if arglist.actor_lstm or arglist.critic_lstm:
                    for agent in self.trainers:
                        agent.reset_lstm()
                if arglist.tracking:
                    for agent in self.trainers:
                        agent.tracker.reset()
                self.episode_step = 0
                self.episode_rewards.append(0)
                for a in self.agent_rewards:
                    a.append(0)
                self.agent_info.append([[]])
            else:
                self.new_episode=False

            # increment global step counter
            self.train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    self.agent_info[-1][i].append(info_n['n'])
                if self.train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(self.agent_info[:-1], fp)
                    break
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None

            # get same episode sampling
            if arglist.sync_sampling:
                inds = [random.randint(0, len(self.trainers[0].replay_buffer._storage)-1) for i in range(arglist.batch_size)]
            else:
                inds = None

            for agent in self.trainers:
                # if arglist.lstm:
                #     agent.preupdate(inds=inds)
                # else:
                agent.preupdate(inds)
            for agent in self.trainers:
                loss = agent.update(self.trainers, self.train_step)
                if loss is None: continue

            # for displaying learned policies
            if arglist.display:
                self.env.render()
                # continue

            # save model, display training output
            if terminal and (len(self.episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=self.saver)
                # print statement depends on whether or not there are adversaries
                if self.num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        self.train_step, len(self.episode_rewards), np.mean(self.episode_rewards[-arglist.save_rate:]), round(time.time()-self.t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        self.train_step, len(self.episode_rewards), np.mean(self.episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in self.agent_rewards], round(time.time()-self.t_start, 3)))
                self.t_start = time.time()
                # Keep track of final episode reward
                self.final_ep_rewards.append(np.mean(self.episode_rewards[-arglist.save_rate:]))
                for rew in self.agent_rewards:
                    self.final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

    def finish(self, arglist):
        # saves final episode reward for plotting training curve later
        # U.save_state(arglist.save_dir, saver=saver)
        if arglist.tracking:
            for agent in self.trainers:
                agent.tracker.save()

        if not os.path.exists("rewards"):
            os.makedirs("rewards")
        rew_file_name = "rewards/" + arglist.commit_num + "_rewards.pkl"
        with open(rew_file_name, 'wb') as fp:
            pickle.dump(self.final_ep_rewards, fp)
        agrew_file_name = "rewards/" + arglist.commit_num + "_agrewards.pkl"
        # agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
        with open(agrew_file_name, 'wb') as fp:
            pickle.dump(self.final_ep_ag_rewards, fp)
        print('...Finished total of {} episodes.'.format(len(self.episode_rewards)))

        self.sess.close()

    def episodes_seen(self):
        return len(self.episode_rewards)

def train(arglist):
    tt = trainTest()
    tt.init(arglist)

    #create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    episode_step = 0

    #get the first observation (rew, done, info are dummies)
    obs_n = env.reset()
    rew_n = False
    done_n = False
    info_n = False
    terminal = False
    print('Starting iterations...')
    while tt.episodes_seen() <= arglist.num_episodes:
        action = tt.update(arglist, obs_n, rew_n, done_n, info_n, terminal)

        # environment step
        obs_n, rew_n, done_n, info_n = env.step(action)
        episode_step += 1

        done = all(done_n)
        terminal =  episode_step >= arglist.max_episode_len

        # reset environment if finished
        if done or terminal:
            obs_n = env.reset()
            episode_step = 0

        # for displaying learned policies
        if arglist.display:
            env.render()

    tt.finish(arglist)



if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
