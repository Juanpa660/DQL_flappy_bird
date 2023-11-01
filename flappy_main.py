#!/usr/bin/env python
import tensorflow as tf
import cv2
import sys
import os
import random
import numpy as np
from collections import deque
sys.path.append("game/")
import wrapped_flappy_bird as game
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.01, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape, stride):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W, stride) + b)


def createCNN():
    c_layer1 = convolutional_layer(input_x=obs_ph, shape=(8, 8, 4, 32), stride=4)
    p_layer1 = max_pool_2by2(c_layer1)
    c_layer2 = convolutional_layer(input_x=p_layer1, shape=(4, 4, 32, 64), stride=2)
    p_layer2 = max_pool_2by2(c_layer2)
    c_layer3 = convolutional_layer(input_x=p_layer2, shape=(3, 3, 64, 64), stride=1)
    p_layer3 = max_pool_2by2(c_layer3)
    flat_layer = tf.reshape(p_layer3, [-1, 256])
    flat_layer2 = tf.layers.dense(inputs=flat_layer, units=256, activation=tf.nn.relu,kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    cnn_output = tf.layers.dense(inputs=flat_layer2, units=2,kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return cnn_output


def img_process(img):
    img_gray = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    img_reshaped = np.reshape(img_bin, [80, 80, 1])
    return img_reshaped


def act_greedy(epsilon, explore_rate, q_value):
    act_out = [0, 0]
    if random.random() <= epsilon:
        act_out[random.randint(0, 1)] = 1
        if epsilon > 0:
            epsilon -= explore_rate
    else:
        act_out_index = np.argmax(q_value)
        act_out[act_out_index] = 1
    return act_out, epsilon


def get_training_set(infos):
    obs_batch = [info[0] for info in infos]
    act_batch = [info[1] for info in infos]
    rew_batch = [info[2] for info in infos]
    obs2_batch = [info[3] for info in infos]
    dones_batch = [info[4] for info in infos]
    return obs_batch, act_batch, rew_batch, obs2_batch, dones_batch


total_memory_len = 50000

""" PLACEHOLDERS """
obs_ph = tf.placeholder(tf.float32, (None, 80, 80, 4), 'obs')
act_ph = tf.placeholder(tf.float32, [None, 2])
q_ph = tf.placeholder(tf.float32, [None])

""" SEEDS """
tf.set_random_seed(1)
np.random.seed(1)
random.seed(1)

def main(epsilon, explore_rate, gamma, learning_rate, obs_timesteps, batch, episodes, writer_name):
    sess = tf.Session()

    """ CREATE CNN """
    q_out = createCNN()

    """ INITIAL FUNCTIONS """
    game_state = game.GameState()
    infos = deque()

    """ COST FUNCTION """
    q_net = tf.reduce_sum(tf.multiply(q_out, act_ph), reduction_indices=1)
    q_loss = tf.reduce_mean(tf.square(q_ph - q_net))
    trainer = tf.train.AdamOptimizer(learning_rate).minimize(q_loss)

    act0 = [1, 0]
    img0, rew_t, done = game_state.frame_step(act0)
    img0_gray = cv2.cvtColor(cv2.resize(img0, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, img0_bin = cv2.threshold(img0_gray, 1, 255, cv2.THRESH_BINARY)
    img0_bin = np.array(img0_bin)
    obs_t = np.stack((img0_bin, img0_bin, img0_bin, img0_bin), axis=2)

    """ SAVER """
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(writer_name, tf.get_default_session())

    episode = 0
    epi_reward = 0
    epi_jumps = 0
    epi_q_max = []
    t = 0
    while True:
        # OBTENGO Q DE LA CNN
        q = sess.run(q_out, feed_dict={obs_ph: [obs_t]})[0]
        # CALCULA ACCION CON E-GREEDY
        act_t, epsilon = act_greedy(epsilon, explore_rate, q)
        # REALIZA ACCION Y SE OBTIENE ESTADO SIGUIENTE
        img, rew_t, done = game_state.frame_step(act_t)
        img_pr = img_process(img)
        obs_t2 = np.append(img_pr, obs_t[:, :, :3], axis=2)

        infos.append((obs_t, act_t, rew_t, obs_t2, done))
        if len(infos) > total_memory_len: infos.popleft()

        if t > obs_timesteps:
            info_sample = random.sample(infos, batch)
            obs_batch, act_batch, rew_batch, obs_t2_batch, dones_batch = get_training_set(info_sample)


            q_batch = []
            q_t2 = sess.run(q_out, feed_dict={obs_ph: obs_t2_batch})
            for i in range(0, len(dones_batch)):
                if dones_batch[i]:
                    q_batch.append(rew_batch[i])
                else:
                    q_batch.append(rew_batch[i] + gamma * np.max(q_t2[i]))

            sess.run(trainer, feed_dict={q_ph: q_batch,
                                   act_ph: act_batch,
                                   obs_ph: obs_batch})

            """ TENSORBOARD STATS """
            epi_reward += rew_t
            epi_jumps += act_t[1]
            epi_q_max.append(np.max(q))
            if done:
                if episode % 20 == 0:
                    print('Episode:', episode, 'Q max mean:', np.mean(epi_q_max))
                episode += 1
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Stats/Episode Reward', simple_value=epi_reward)]), episode)
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Stats/Episode Jumps', simple_value=epi_jumps)]), episode)
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Stats2/Epsilon', simple_value=epsilon)]),episode)
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Stats/Q max', simple_value=np.mean(epi_q_max))]), episode)
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Stats2/Q Loss',simple_value=sess.run(q_loss,
                feed_dict={q_ph: q_batch,act_ph: act_batch,obs_ph: obs_batch}))]),episode)

                epi_q_max = []
                epi_reward = 0
                epi_jumps = 0

                if episode > episodes:
                    saver.save(sess, 'saved_network/flappy-dqn', global_step=episode)
                    break

        obs_t = obs_t2
        t+=1
    writer.close()
    sess.close()



if __name__ == "__main__":
    main(epsilon=0.1, explore_rate=10e-7, gamma=0.99, learning_rate=10e-7,
         obs_timesteps=10000, batch=32, episodes=25000, writer_name='./log/Flappy_new1')
