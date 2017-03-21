#!/usr/bin/env python
import argparse
import logging
import gym
import sys
import universe
from universe import pyprofile, wrappers

from GameSettingsEvent import GTASetting
import time

# agent includes
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from pos import current_state
from OU import OU

# ip of computer running GTA
remote_ip = '128.237.99.169'

# default is 5900 but some networks may block outside access
# in that case, we can configure the vnc server on a different port
vnc_port = '8989'

websocket_port = '15900'

# create gym-like environment
env = wrappers.WrappedVNCEnv()

# The GymCoreSyncEnv's try to mimic their core counterparts,
# and thus came pre-wrapped wth an action space
# translator. Everything else probably wants a SafeActionSpace
# wrapper to shield them from random-agent clicking around
# everywhere.
env = wrappers.experimental.SafeActionSpace(env)

env.configure(
    fps=8,
    # print_frequency=None,
    # ignore_clock_skew=True,
    remotes='vnc://'+remote_ip+':'+vnc_port+'+'+websocket_port,
    vnc_driver='go', vnc_kwargs={
        'encoding': 'tight', 'fine_quality_level': 0, 'subsample_level': 3, 'quality_level': 0,
    },
)

# show VNC window?
render = False


### SET UP AGENT

train_indicator = 1 # 1 if training

OU = OU()  # Ornstein-Uhlenbeck Process
BUFFER_SIZE = 100000
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.1  # Target Network HyperParameter
LRA = 0.0001  # Learning rate for Actor
LRC = 0.001  # Lerning rate for Critic

destination_pos = 7 #todo: change
av_pos = -10 # todo
av_xpos = 2 # todo
av_angle = np.pi / 2
dist_t0, area_t0 = current_state(av_pos)

l1 = 8
l2 = 8
l3 = 8
l4 = 8
v_t = 0
r_t_area = 1


action_dim = 1 #3  # Steering/Acceleration/Brake

state_dim = 25 #29  # of sensors input

np.random.seed(1337)

EXPLORE = 100000.
episode_count = 2000
max_steps = 100000
reward = 0
done = False
step = 0
epsilon = 1

# create TF session
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
buff = ReplayBuffer(BUFFER_SIZE)


observation_n = env.reset()


reward_n = [0] * env.n
done_n = [False] * env.n
info = None

for i in range(episode_count):

    print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))


    # get state from environment
    dist_t, area_t = current_state(av_pos)

    # form full state for input to network
    s_t = np.hstack((np.array([l1, l2, l3, l4, av_xpos, av_pos, av_angle], ndmin=2), dist_t))

    total_reward = 0.
    for j in range(max_steps):
        loss = 0

        # anneal our exploration rate
        epsilon -= 1.0 / EXPLORE

        a_t = np.zeros([1, action_dim])
        noise_t = np.zeros([1, action_dim])

        a_t_original = actor.model.predict(s_t)
        noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)

        a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
        v_t_ori = v_t

        # no negative velocity
        v_t = np.maximum(v_t + a_t[0][0] * TAU, 0)



            # ob, r_t, done, info = env.step(a_t[0])
            r_t_ori = r_t_area

            r_t_area = area_t/area_t0
            av_pos += 0.5 * (v_t_ori + v_t) *TAU
            
            if r_t_area < r_t_ori:
                r_t = - r_t_area
            else:
                r_t = r_t_area
            # if a_t[0][0] > 0:
            #     av_pos = destination_pos
            #     # r_t += 1
            # else:
            #     av_pos += 0.5 * 1 * (TAU ** 2)
            #     r_t -= 0.5
            if av_pos >= destination_pos:
                done = True

            dist_t, area_t = current_state(av_pos)
            # s_t1 = np.hstack(
                # (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            s_t1 = np.hstack((np.array([l1, l2, l3, l4, av_xpos, av_pos, av_angle], ndmin=2), dist_t))

            buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add to replay buffer

            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.squeeze(np.asarray([e[0] for e in batch]), axis=1)
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.squeeze(np.asarray([e[3] for e in batch]), axis=1)
            dones = np.asarray([e[4] for e in batch])
            # y_t = np.asarray([e[1] for e in batch])
            y_t = rewards


            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                v_t = 0
                av_pos = -10
                done = False
                r_t_area = 1
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    # env.end()  # This is for shutting down TORCS
    print("Finish.")
for i in range(args.max_steps):
    if render:
        # Note the first time you call render, it'll be relatively
        # slow and you'll have some aggregated rewards. We could
        # open the render() window before `reset()`, but that's
        # confusing since it pops up a black window for the
        # duration of the reset.
        env.render()

    action_n = driver.step(observation_n, reward_n, done_n, info)

    #print observation_n
    #env.reset()
    #time.sleep(1)

    try:
        if info is not None:
            distance = info['n'][0]['distance_from_destination']
            logger.info('distance %s', distance)
    except KeyError as e:
        logger.debug('distance not available %s', str(e))

    # if args.custom_camera:
    #     # Sending this every step is probably overkill
    #     for action in action_n:
    #         action.append(GTASetting('use_custom_camera', True))

    # Take an action
    with pyprofile.push('env.step'):
        _step = env.step(action_n)
        observation_n, reward_n, done_n, info = _step

    if info is not None:
        try:
            x = info['n'][0]['x_coord']
            y = info['n'][0]['y_coord']
            z = info['n'][0]['z_coord']

            scans = info['n'][0]['scans']

            print "got scans"

            scan_data = np.zeros((len(scans)+1, 2))
            scan_data[0,:] = [x, y]

            for i in range(len(scans)):
                scan_data[i+1, :] = [scans[i]['x'], scans[i]['y']]

            with data_q.mutex:
                # clear the list
                data_q.queue[:] = []

            print "adding some data"
            data_q.put(scan_data)

        except Exception as e:
            print e

    if any(done_n) and info and not any(info_n.get('env_status.artificial.done', False) for info_n in info['n']):
        print('done_n', done_n, 'i', i)
        logger.info('end of episode')
        env.reset()

# We're done! clean up
env.close()


