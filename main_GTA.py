#!/usr/bin/env python
import argparse
import logging
import gym
import sys
import numpy as np
import universe
from universe import pyprofile, wrappers
from universe.spaces.joystick_event import JoystickAxisXEvent, JoystickAxisZEvent

from utils.GameSettingsEvent import GTASetting
import time

# agent includes
import tensorflow as tf
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from pos import current_state
from OU import OU
import json

# send "no-operation" action to environment
def get_noop():
    x_axis_event = JoystickAxisXEvent(0)
    z_axis_event = JoystickAxisZEvent(0)
    noop = [[x_axis_event, z_axis_event]]
    return noop
noop_action = get_noop()

def scans2dists(info):
    x = info['n'][0]['x_coord']
    y = info['n'][0]['y_coord']
    car_pos = np.array([x, y])

    scans = info['n'][0]['scans']
    scan_data = np.zeros((len(scans), 2))

    for i in range(len(scans)):
        scan_data[i, :] = [scans[i]['x'], scans[i]['y']]

    dists = np.sqrt(np.sum((car_pos-scan_data)**2,axis=1))

    area = sum(1./len(scans) * np.pi * (dists**2))

    return np.array([dists]), area

def get_dest_dist(info):
    x = info['n'][0]['x_coord']
    y = info['n'][0]['y_coord']
    z = info['n'][0]['z_coord']
    car_pos = np.array([x, y, z])
    destination_pos = np.array([1776.372, 3706.432, 33.8789])
    dest_dist = np.sqrt(np.sum((car_pos-destination_pos)**2))
    return dest_dist

# send no-ops until we get valid state info
def synchronous_step(env, action):

    # first, send the real action
    # then wait until we get a response with state info
    # or should we keep sending the same action??

    while True:
        observation_n, reward_n, done_n, info = env.step(action)

        try:
            # if we can access the car position, the rest of the info should be there?
            _ = info['n'][0]['x_coord']
            break
        except Exception as e:
            pass

        # may need to catch done signal
        if any(done_n) and info and not any(info_n.get('env_status.artificial.done', False) for info_n in info['n']):
            env.reset()
        time.sleep(0.1)

    return observation_n, reward_n, done_n, info

# ip of computer running GTA
remote_ip = '128.237.99.169'

# default is 5900 but some networks may block outside access
# in that case, we can configure the vnc server on a different port
vnc_port = '8989'

websocket_port = '15900'

# create gym-like environment
env = gym.make('gtav.SaneDriving-v0')

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

state_dim = 30 #29  # of sensors input

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

noop_action = get_noop()

for i in range(episode_count):

    step = 0
    print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))


    # since environment is asynchronous, wait until we get a valid state
    # send no-ops until we are valid
    #time.sleep(3)
    _, _, _, info = synchronous_step(env, noop_action)

    # compute distances from scan locations
    dist_t, area_t = scans2dists(info)
    #area_t0 = area_t
    # todo: make av_pos distance from goal?
    av_pos = get_dest_dist(info)

    # compose full state for input to network
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


        # compute reward of the area captured by "linescans"
        r_t_ori = r_t_area
        r_t_area = area_t/area_t0
        if r_t_area < r_t_ori:
            r_t = - r_t_area
        else:
            r_t = r_t_area


        # send action to environment
        # av_pos += 0.5 * (v_t_ori + v_t) *TAU

        action_n = [[GTASetting('set_velocity', v_t)]]
        with pyprofile.push('env.step'):
            observation_n, reward_n, done_n, info = synchronous_step(env, action_n)


        # compose full state at t+1 and save to buffer
        dist_t, area_t = scans2dists(info)
        av_pos = get_dest_dist(info)

        s_t1 = np.hstack((np.array([l1, l2, l3, l4, av_xpos, av_pos, av_angle], ndmin=2), dist_t))

        buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add to replay buffer

        # sample a batch from the replay buffer
        batch = buff.getBatch(BATCH_SIZE)
        states = np.squeeze(np.asarray([e[0] for e in batch]), axis=1)
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.squeeze(np.asarray([e[3] for e in batch]), axis=1)
        dones = np.asarray([e[4] for e in batch])
        # y_t = np.asarray([e[1] for e in batch])
        y_t = rewards


        # update networks using batch
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

        print("Episode", i, "Step", step, "Action", (a_t, v_t), "Reward", r_t, "Loss", loss)

        step += 1

        # check if environment is done
        # hack for now: reset on agent side; should ideally happen on environment side
        # because updates come slowly over the connection; we might pass our goal position

        dest_dist = av_pos
        print ("DISTANCE TO DEST: ", dest_dist)

        # meters?
        if dest_dist <= 5:
            v_t = 0
            r_t_area = 1
            env.reset()
            break


        if any(done_n) and info and not any(info_n.get('env_status.artificial.done', False) for info_n in info['n']):
            print "END OF EPISODE"
            env.reset()

            v_t = 0
            r_t_area = 1

            break

    # save model parameters every few episodes
    if np.mod(i, 10) == 0:
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


# We're done! clean up
env.close()
print("Finish.")


