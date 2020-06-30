class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer

        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

        # set up tracking
        self.tracker = InfoTracker(self.name, self.args)

    def action(self, obs):
        action = self.act(obs[None])[0]
        if self.args.tracking: self.tracker.record_information("communication", np.argmax(action[0][-2:]))
        return action

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.tracker.start()

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            rew = np.reshape(rew, target_q_next.shape)
            done = np.reshape(done, target_q_next.shape)
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next

        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        if self.args.tracking:
            self.tracker.record_information("q_loss",q_loss)
            self.tracker.record_information("p_loss", p_loss)
            self.tracker.record_information("target_q_mean", np.mean(target_q))
            self.tracker.record_information("reward_mean", np.mean(rew))
            self.tracker.record_information("target_q_next_mean", np.mean(target_q_next))
            self.tracker.record_information("target_q_std", np.std(target_q))


        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]


def p_LSTM_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func,
            optimizer, grad_norm_clipping=None, local_q_func=False,
            num_units=64, scope="trainer", reuse=None, q_debug=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None, 1], name="action"+str(i)) for i in range(len(act_space_n))]
        act_logits_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action_mode" + str(i)) for i in range(len(act_space_n))]

        q_res = 1
        p_res = int(act_pdtype_n[p_index].param_shape()[0])

        # for actor
        p_c_ph, p_h_ph = get_lstm_state_ph(name='p_', n_batches=None, num_units=num_units)
        p_c_ph_n, p_h_ph_n = [p_c_ph for i in range(len(obs_ph_n))], [p_h_ph for i in range(len(obs_ph_n))]
        # for critic
        q_c_ph, q_h_ph = get_lstm_state_ph(name='q_', n_batches=None, num_units=num_units)
        q_c_ph_n, q_h_ph_n = [q_c_ph for i in range(len(obs_ph_n))], [q_h_ph for i in range(len(obs_ph_n))]

        p_input = tf.concat([obs_ph_n[p_index], p_c_ph, p_h_ph], -1)
        # p_input = obs_ph_n[p_index]
        p, p_state_out = p_func(p_input, p_res, scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()

        # need to check this -- need safety checks
        q_input = tf.concat(obs_ph_n + act_input_n + q_c_ph_n + q_h_ph_n, -1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], -1)
        q, q_state_out= q_func(q_input, 1, scope="q_func", num_units=num_units, reuse=True)

        q = q[:,0]

        # testing:
        q_values = U.function(inputs=obs_ph_n + act_ph_n + q_c_ph_n + q_h_ph_n, outputs=[q, q_state_out])

        pg_loss = -tf.reduce_mean(q)
        q_values = U.function(inputs=obs_ph_n + act_ph_n + q_c_ph_n + q_h_ph_n, outputs=q)


        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + q_c_ph_n + q_h_ph_n + p_c_ph_n + p_h_ph_n, outputs=loss, updates=[optimize_expr])
        # act = U.function(inputs=[obs_ph_n[p_index], p_c_ph, p_h_ph], outputs=act_sample)
        act = U.function(inputs=[obs_ph_n[p_index], p_c_ph, p_h_ph], outputs=[act_sample, p_state_out])
        p_values = U.function(inputs=[obs_ph_n[p_index], p_c_ph, p_h_ph], outputs=p)

        # target network
        target_p, target_p_state_out = p_func(p_input, p_res, scope="target_p_func", num_units=num_units)


        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_pd = act_pdtype_n[p_index].pdfromflat(target_p)
        target_act_sample = target_pd.sample()

        target_act = U.function(inputs=[obs_ph_n[p_index], p_c_ph, p_h_ph], outputs=target_act_sample)

        target_ph_pd = act_pdtype_n[p_index].pdfromflat(act_logits_ph_n[p_index])
        kl_move_loss = tf.reduce_mean(target_pd.kl(target_ph_pd))
        f_kl_move_loss = U.function(inputs=[obs_ph_n[p_index],act_logits_ph_n[p_index]],outputs=kl_move_loss)
        kl_comm_loss = tf.reduce_mean(target_pd.kl_comm(target_ph_pd))
        f_kl_comm_loss = U.function(inputs=[obs_ph_n[p_index],act_logits_ph_n[p_index]],outputs=kl_comm_loss)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act, 'testing':q_values, 'f_kl_move_loss':f_kl_move_loss, 'f_kl_comm_loss':f_kl_comm_loss}

def q_LSTM_train(make_obs_ph_n, act_space_n, q_index, q_func,
            optimizer, grad_norm_clipping=None, local_q_func=False,
            scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None, 1], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None, 1], name="target")

        q_res = 1
        q_c_ph, q_h_ph = get_lstm_state_ph(name='q_', n_batches=None, num_units=num_units)
        q_c_ph_n, q_h_ph_n = [q_c_ph for i in range(len(obs_ph_n))], [q_h_ph for i in range(len(obs_ph_n))]
        # need to check this -- need safety checks
        # q_input = tf.concat(obs_ph_n + act_ph_n, -1)
        q_input = tf.concat(obs_ph_n + act_ph_n + q_c_ph_n + q_h_ph_n, -1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], -1)

        q, q_state_out = q_func(q_input, 1, scope="q_func", num_units=num_units)
        q = q[:,0]

        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        q_values = U.function(inputs=obs_ph_n + act_ph_n + q_c_ph_n + q_h_ph_n, outputs=[q, q_state_out])
        train = U.function(inputs=obs_ph_n + act_ph_n + q_c_ph_n + q_h_ph_n+ [target_ph], outputs=loss, updates=[optimize_expr])

        # target network
        target_q, target_q_state_out = q_func(q_input, 1, scope="target_q_func", num_units=num_units)
        target_q = target_q[:,0]

        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(inputs=obs_ph_n + act_ph_n + q_c_ph_n + q_h_ph_n, outputs=target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}
