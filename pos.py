#!/usr/bin/env python
from intersectionBoundary import drawIntersection
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

pos_x = 2
av_pos = -6
n_degree = 18
gamma= 0.9
learning_rate = 0.01
display_step = 10
tau = 0.01
vel = 2

# Network Parameters
n_hidden_1 = 300  # 1st layer number of features
n_hidden_2 = 600  # 2nd layer number of features
state_size = n_degree
p_size = 1
action_size = 1
reward_size = 1


def linetrace(pos, n):
    r1 = 2 - (-4)
    r2 = 4 - pos
    r3 = 4 - 2
    rmax = 80
    theta = np.arange(0.001, np.pi, np.pi/n)
    dist = []
    x = 0
    y = 0
    for i in range(0, len(theta)):
        if pos <= -4:
            if theta[i] <= np.arctan(((-4) - pos)/(pos_x-(-4))):
                dist.append(np.minimum(r1/np.cos(theta[i]),rmax))
                x = np.arange(pos_x - dist[i] * np.cos(theta[i]), pos_x, 0.01)
                y = - (x - pos_x) * np.tan(theta[i]) + pos
            else:
                if theta[i] > np.arctan(((-4) - pos)/(pos_x-(-4))) and theta[i] <= np.arctan(((4) - pos)/(pos_x-(-4))):
                    dist.append(np.minimum(r2/np.sin(theta[i]),rmax))
                    x = np.arange(pos_x - dist[i] * np.cos(theta[i]), pos_x, 0.01)
                    y = - (x - pos_x) * np.tan(theta[i]) + pos
                else:
                    if theta[i] > np.arctan(((4) - pos)/(pos_x-(-4))) and theta[i] <= np.pi/2:
                        dist.append(np.minimum(r1 / np.cos(theta[i]),rmax))
                        x = np.arange(pos_x - dist[i] * np.cos(theta[i]), pos_x, 0.01)
                        y = - (x - pos_x) * np.tan(theta[i]) + pos
                    else:
                        if theta[i] > np.pi/2 and theta[i] <= np.pi/2 + np.arctan(4 - pos_x)/(4 - pos):
                            dist.append(np.minimum(r3/np.sin(theta[i] - np.pi/2),rmax))
                            x = np.arange(pos_x, pos_x - dist[i] * np.cos(theta[i]), 0.01)
                            y = - (x - pos_x) * np.tan(theta[i]) + pos
                        else:
                            if theta[i] > np.pi/2 + np.arctan(4 - pos_x)/(4 - pos) and theta[i] < np.pi/2 + np.arctan((4 - pos_x)/(-4 - pos)):
                                dist.append(np.minimum(r2 / np.cos(theta[i]-np.pi/2),rmax))
                                x = np.arange(pos_x, pos_x - dist[i] * np.cos(theta[i]), 0.01)
                                y = - (x - pos_x) * np.tan(theta[i]) + pos
                            else:
                                dist.append(np.minimum(r3 / np.sin(theta[i] - np.pi/2),rmax))
                                x = np.arange(pos_x, pos_x - dist[i] * np.cos(theta[i]), 0.01)
                                y = - (x - pos_x) * np.tan(theta[i]) + pos
        else:
            if theta[i] <= np.arctan(((4) - pos) / (pos_x - (-4))):
                dist.append(np.minimum(r2 / np.sin(theta[i]), rmax))
                x = np.arange(pos_x - dist[i] * np.cos(theta[i]), pos_x, 0.01)
                y = - (x - pos_x) * np.tan(theta[i]) + pos
            else:
                if theta[i] > np.arctan(((4) - pos) / (pos_x - (-4))) and theta[i] <= np.pi / 2:
                    dist.append(np.minimum(r1 / np.cos(theta[i]), rmax))
                    x = np.arange(pos_x - dist[i] * np.cos(theta[i]), pos_x, 0.01)
                    y = - (x - pos_x) * np.tan(theta[i]) + pos
                else:
                    if theta[i] > np.pi / 2 and theta[i] <= np.pi / 2 + np.arctan(4 - 2) / (4 - pos):
                        dist.append(np.minimum(r3 / np.sin(theta[i] - np.pi / 2), rmax))
                        x = np.arange(pos_x, pos_x - dist[i] * np.cos(theta[i]), 0.01)
                        y = - (x - pos_x) * np.tan(theta[i]) + pos
                    else:
                        dist.append(np.minimum(r2 / np.cos(theta[i] - np.pi / 2), rmax))
                        x = np.arange(pos_x, pos_x - dist[i] * np.cos(theta[i]), 0.01)
                        y = - (x - pos_x) * np.tan(theta[i]) + pos
        plt.plot(x, y, 'g')
        plt.draw()
    return dist


def dis_function(pos, n):
    drawIntersection()
    state_dis = linetrace(pos, n)
    plt.plot(pos_x, pos, 'r', marker='s',markersize=8)
    plt.draw()
    return state_dis


def get_state(pos, n):
    # x0 = 2*np.random.rand(1)
    state_dis = dis_function(pos, n)
    return state_dis #np.array(np.concatenate([x0, state_dis]), ndmin=2)


def get_next_state(pos, a, n):
    if a == 0:
        pos = tf.add(pos, 0.1*2)
        s = get_state(pos, n)
    else:
        pos = 4
        s = get_state(pos, n)
    return pos, s


def s_reward(s, n):
    return sum(1./n * np.pi * (s[0,:]**2))


def all_reward(s_, s, a, n):
    if s_reward(s_, n) < s_reward(s, n):
        reward = 1
    else:
        reward = -1
    if a == 1:
        reward += 5
    return reward


def current_state(pos):
    plt.ion()
    new_dis = np.array(get_state(pos, n_degree), ndmin=2)

    new_area = s_reward(new_dis, n_degree)

    plt.pause(0.01)
    plt.clf()
    return new_dis, new_area


def state_table(av_pos):
    plt.ion()
    state = np.array(get_state(av_pos, n_degree), ndmin=2)
    state_reward = [s_reward(state, n_degree)]
    pos = av_pos
    i = 1
    while av_pos <= 4:
        av_pos += tau * vel
        new_state = np.array(get_state(av_pos, n_degree), ndmin=2)
        new_reward = s_reward(new_state, n_degree)
        # print(new_reward)
        # print(state_reward)
        state = np.vstack((state, new_state))
        state_reward = np.vstack((state_reward, new_reward))
        pos = np.vstack((pos, av_pos))
        # plt.show()
        plt.pause(0.01)
        plt.clf()
        i += 1
    return pos, state, state_reward

# pos, state, state_reward = state_table(av_pos)