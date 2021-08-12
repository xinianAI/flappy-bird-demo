import os
import pickle

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import random
from collections import deque
import wrapped_flappy_bird as game
import cv2
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100000.  # timesteps to observe before training
EXPLORE = 2000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # 0.001 # final value of epsilon
INITIAL_EPSILON = 0.1  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 500
RECORD_STEP = (500000, 1000000, 1500000, 2000000, 2500000, 3000000)    # the time steps to draw pics.
url2 = 'D:/flapPyBirdDemo/game/'
gamename = 'bird'

try:
    tf.mul
except:
    # For new version of tensorflow
    # tf.mul has been removed in new version of tensorflow
    # Using tf.multiply to replace tf.mul
    tf.mul = tf.multiply


class BrainDQN:

    def __init__(self, actions):
        self.gameName = gamename
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.onlineTimeStep = 0
        self.timeStep = 0
        self.score = 0
        # saved parameters every SAVER_ITER
        self.gameTimes = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        self._setDirName()
        # logs, append to file every SAVER_ITER
        self.save_path = url2 + "/saved_parameters" + self.dir_name  # store network parameters and other parameters for pause.
        self.saved_parameters_file_path = self.save_path + self.gameName + '-saved-parameters.txt'
        self.logs_path = "./logs_" + self.gameName + self.dir_name  # "logs_bird/dqn/"

        self.lost_hist = []
        self.lost_hist_file_path = self.logs_path + 'lost_hist.txt'
        self.q_target_list = []
        self.q_target_file_path = self.logs_path + 'q_targets.txt'
        self.score_every_episode = []
        self.score_every_episode_file_path = self.logs_path + 'score_every_episode.txt'
        self.time_steps_when_episode_end = []
        self.time_steps_when_episode_end_file_path = self.logs_path + 'time_steps_when_episode_end.txt'
        self.reward_every_time_step = []
        self.reward_every_time_step_file_path = self.logs_path + 'reward_every_time_step.txt'
        # load network and other parameters

        # init Q network
        self.stateInput, self.QValue, self.W_conv1_1, self.b_conv1_1, self.W_conv2_1, self.b_conv2_1, self.W_conv2_2, \
        self.b_conv2_2, self.W_conv3_1, self.b_conv3_1, self.W_conv3_2, self.b_conv3_2,  \
        self.W_fc1, self.b_fc1, self.W_fc2_v, self.b_fc2_v, self.W_fc2_a, self.b_fc2_a, self.h_fc1 = self.createQNetwork()

        # init Target Q Network
        self.stateInputT, self.QValueT, self.W_conv1_1T, self.b_conv1_1T, self.W_conv2_1T, self.b_conv2_1T, \
        self.W_conv2_2T, self.b_conv2_2T, self.W_conv3_1T, self.b_conv3_1T, self.W_conv3_2T, self.b_conv3_2T, \
        self.W_fc1T, self.b_fc1T, self.W_fc2T_v, self.b_fc2T_v, self.W_fc2T_a, self.b_fc2T_a, \
        self.h_fc1T = self.createQNetwork()

        self.copyTargetQNetworkOperation = [self.W_conv1_1T.assign(self.W_conv1_1),
                                            self.b_conv1_1T.assign(self.b_conv1_1),
                                            self.W_conv2_1T.assign(self.W_conv2_1),
                                            self.b_conv2_1T.assign(self.b_conv2_1),
                                            self.W_conv2_2T.assign(self.W_conv2_2),
                                            self.b_conv2_2T.assign(self.b_conv2_2),
                                            self.W_conv3_1T.assign(self.W_conv3_1),
                                            self.b_conv3_1T.assign(self.b_conv3_1),
                                            self.W_conv3_2T.assign(self.W_conv3_2),
                                            self.b_conv3_2T.assign(self.b_conv3_2),
                                            self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc2T_v.assign(self.W_fc2_v), self.b_fc2T_v.assign(self.b_fc2_v),
                                            self.W_fc2T_a.assign(self.W_fc2_a), self.b_fc2T_a.assign(self.b_fc2_a)]
        self.createTrainingMethod()
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self._load_saved_parameters()
        # self._load_saved_parameters()
        # Evaluation: store the last ten episodes' scores
        self.counters = []
        # tensorboard
        tf.summary.FileWriter(self.logs_path, self.sess.graph)

    def _setDirName(self):
        self.dir_name = "/dqn_Dueling/"

    def createQNetwork(self):
        # 输入层
        # 定义80*80*4的输入层s（处理过的连续4帧的游戏图像）
        s = tf.placeholder('float', [None, 80, 80, 4])

        # 隐藏层

        # 在达到相同感受野的情况下，卷积核越小，所需要的参数和计算量越小
        # 卷积 池化过程
        w_conv1_1 = self.get_weight([3, 3, 4, 16])  # 第一个卷积核 3*3 4通道 16核
        b_conv1_1 = self.get_bias([16])  # 偏置b
        # 卷积
        h_conv1_1 = tf.nn.relu(self.conv2d(s, w_conv1_1, 1) + b_conv1_1)
        # 池化
        h_pool1 = self.max_pool_2x2(h_conv1_1)

        # 连续两个卷积+一个池化过程
        w_conv2_1 = self.get_weight([3, 3, 16, 32])
        b_conv2_1 = self.get_bias([32])
        # 卷积2_1
        h_conv2_1 = tf.nn.relu(self.conv2d(h_pool1, w_conv2_1, 1) + b_conv2_1)
        w_conv2_2 = self.get_weight([3, 3, 32, 32])
        b_conv2_2 = self.get_bias([32])
        h_conv2_2 = tf.nn.relu(self.conv2d(h_conv2_1, w_conv2_2, 1) + b_conv2_2)
        # 池化
        h_pool2 = self.max_pool_2x2(h_conv2_2)

        # 连续三个卷积+一个池化过程
        w_conv3_1 = self.get_weight([3, 3, 32, 64])
        b_conv3_1 = self.get_bias([64])
        # 卷积3_1
        h_conv3_1 = tf.nn.relu(self.conv2d(h_pool2, w_conv3_1, 1) + b_conv3_1)
        w_conv3_2 = self.get_weight([3, 3, 64, 64])
        b_conv3_2 = self.get_bias([64])
        h_conv3_2 = tf.nn.relu(self.conv2d(h_conv3_1, w_conv3_2, 1) + b_conv3_2)
        # 池化
        h_pool3 = self.max_pool_2x2(h_conv3_2)  # 10*10*64

        # 扁平化
        # tf.reshape 函数原型为
        # def reshape(tensor, shape, name=None)
        # 第1个参数为被调整维度的张量 第2个参数为要调整为的形状
        # 返回一个shape形状的新tensor
        # 注意shape里最多有一个维度的值可以填写为 - 1，表示自动计算此维度
        h_conv3_flat = tf.reshape(h_pool3, [-1, 6400])  # 1*6400

        # 全连接层 得到小鸟每个动作对应的Q值out
        # 第一层
        w_fc1 = self.get_weight([6400, 512])
        b_fc1 = self.get_bias([512])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1)

        # 输出层
        w_fc2_v = self.get_weight([512, 1])
        b_fc2_v = self.get_bias([1, 1])
        eval_V = tf.matmul(h_fc1, w_fc2_v) + b_fc2_v
        w_fc2_a = self.get_weight([512, self.actions])
        b_fc2_a = self.get_bias([1, self.actions])
        eval_A = tf.matmul(h_fc1, w_fc2_a) + b_fc2_a

        readout = eval_V + (eval_A - tf.reduce_mean(eval_A, axis=1, keep_dims=True))

        return s, readout, w_conv1_1, b_conv1_1, w_conv2_1, b_conv2_1, w_conv2_2, b_conv2_2, w_conv3_1, b_conv3_1, w_conv3_2, b_conv3_2,  w_fc1, b_fc1, w_fc2_v, b_fc2_v, w_fc2_a, b_fc2_a, h_fc1

    def copyTargetQNetwork(self):
        self.sess.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float", [None])
        Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def trainQNetwork(self):
        url = 'D:/flapPyBirdDemo/'
        url2 = 'D:/flapPyBirdDemo/game/'
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextState_batch})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        _, self.lost = self.sess.run(
            [self.trainStep, self.cost],
            feed_dict={
                self.yInput: y_batch,
                self.actionInput: action_batch,
                self.stateInput: state_batch
        })
        self.lost_hist.append(self.lost)
        self.q_target_list.append(y_batch)
        # save network and other data every 100,000 iteration
        if self.timeStep % 100000 == 0:
            self.saver.save(self.sess, self.save_path + self.gameName, global_step=self.timeStep)
            saved_parameters_file = open(self.saved_parameters_file_path, 'wb')
            pickle.dump(self.gameTimes, saved_parameters_file)
            pickle.dump(self.timeStep, saved_parameters_file)
            pickle.dump(self.epsilon, saved_parameters_file)
            pickle.dump(self.replayMemory, saved_parameters_file)
            saved_parameters_file.close()
            self._save_loss_score_timestep_reward_qtarget_to_file()
        if self.timeStep in RECORD_STEP:
            self._record_by_pic()

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

        return QValue_batch

    def setPerception(self, nextObservation, action, reward, terminal, curScore):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        newState = np.append(self.currentState[:, :, 1:], nextObservation, axis=2)
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()
        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"

        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.timeStep, "/ EPSILON", self.epsilon, "/ STATE", state, "/ ACTION", action[1],
              "/ REWARD", reward, "/ SCORE", curScore)
        if not terminal:
            self.score = curScore
        self.reward_every_time_step.append(reward)
        if terminal:
            self.gameTimes += 1
            print("GAME_TIMES:" + str(self.gameTimes))
            print("/ SCORE", self.score)
            self.score_every_episode.append(self.score)
            self.score = 0
            self.time_steps_when_episode_end.append(self.timeStep)
        self.currentState = newState
        self.timeStep += 1
        self.onlineTimeStep += 1

    def getAction(self):
        QValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState]})[0]
        action = np.zeros(self.actions)
        action_index = 0
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def setInitState(self, observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis=2)
        # Called at the record times.

    # load network and other parameters every SAVER_ITER
    def _load_saved_parameters(self):
        print(self.save_path)
        checkpoint = tf.train.get_checkpoint_state(self.save_path)
        print(checkpoint)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            # restore other params.
            if os.path.exists(self.saved_parameters_file_path) and os.path.getsize(self.saved_parameters_file_path) > 0:
                with open(self.saved_parameters_file_path, 'rb') as saved_parameters_file:
                    self.gameTimes = pickle.load(saved_parameters_file)
                    print(self.gameTimes)
                    self.timeStep = pickle.load(saved_parameters_file)
                    print(self.timeStep)
                    self.epsilon = pickle.load(saved_parameters_file)
                    print(self.epsilon)
                    self.replayMemory = pickle.load(saved_parameters_file)
        else:
            # Re-train the network from zero.
            print("Could not find old network weights")

    def _record_by_pic(self):
        self._save_loss_score_timestep_reward_qtarget_to_file()
        loss, scores, time_step_when_episode_end, reward_every_time_step, q_target = self._get_loss_score_timestep_reward_qtarget_from_file()
        plt.figure()
        plt.plot(loss, '-')
        plt.ylabel('loss')
        plt.xlabel('time_step')
        plt.savefig(self.logs_path + str(self.timeStep) + "_lost_hist_total.png")

        plt.figure()
        plt.plot(reward_every_time_step, '-')
        plt.ylabel('reward')
        plt.xlabel('time_step')
        plt.savefig(self.logs_path + str(self.timeStep) + "reward_every_time_step.png")

        plt.figure()
        plt.plot(scores, '-')
        plt.ylabel('score')
        plt.xlabel('episode')
        plt.savefig(self.logs_path + str(self.timeStep) + "_scores_episode_total.png")

        plt.figure()
        plt.plot(q_target, '-')
        plt.ylabel('q_target')
        plt.xlabel('BATCH * time_step')
        plt.savefig(self.logs_path + str(self.timeStep) + "_q_target_total.png")

        plt.figure()
        plt.plot(time_step_when_episode_end, scores, '-')
        plt.ylabel('score')
        plt.xlabel('time_step')
        plt.savefig(self.logs_path + str(self.timeStep) + "_scores_time_step_total.png")

        # save loss/score/time_step/reward/q_target to file

    def _save_loss_score_timestep_reward_qtarget_to_file(self):
        with open(self.lost_hist_file_path, 'a') as lost_hist_file:
            for l in self.lost_hist:
                lost_hist_file.write(str(l) + ' ')
        del self.lost_hist[:]

        with open(self.score_every_episode_file_path, 'a') as score_every_episode_file:
            for s in self.score_every_episode:
                score_every_episode_file.write(str(s) + ' ')
        del self.score_every_episode[:]

        with open(self.time_steps_when_episode_end_file_path, 'a') as time_step_when_episode_end_file:
            for t in self.time_steps_when_episode_end:
                time_step_when_episode_end_file.write(str(t) + ' ')
        del self.time_steps_when_episode_end[:]

        with open(self.reward_every_time_step_file_path, 'a') as reward_every_time_step_file:
            for r in self.reward_every_time_step:
                reward_every_time_step_file.write(str(r) + ' ')
        del self.reward_every_time_step[:]

        with open(self.q_target_file_path, 'a') as q_target_file:
            for q in self.q_target_list:
                q_target_file.write(str(q) + ' ')
        del self.q_target_list[:]

    def get_loss_score_timestep_reward_qtarget_from_file(self):
        # with open(self.lost_hist_file_path, 'r') as lost_hist_file:
        #     lost_hist_list_str = lost_hist_file.readline().split(" ")
        #     lost_hist_list_str = lost_hist_list_str[0:-1]
        #     loss = list(map(eval, lost_hist_list_str))
        #
        # with open(self.score_every_episode_file_path, 'r') as score_every_episode_file:
        #     scores_str = score_every_episode_file.readline().split(" ")
        #     scores_str = scores_str[0:-1]
        #     scores = list(map(eval, scores_str))
        #
        # with open(self.time_steps_when_episode_end_file_path, 'r') as time_step_when_episode_end_file:
        #     time_step_when_episode_end_str = time_step_when_episode_end_file.readline().split(" ")
        #     time_step_when_episode_end_str = time_step_when_episode_end_str[0:-1]
        #     time_step_when_episode_end = list(map(eval, time_step_when_episode_end_str))
        #
        # with open(self.reward_every_time_step_file_path, 'r') as reward_every_time_step_file:
        #     reward_every_time_step_str = reward_every_time_step_file.readline().split(" ")
        #     reward_every_time_step_str = reward_every_time_step_str[0:-1]
        #     reward_every_time_step = list(map(eval, reward_every_time_step_str))
        #
        # with open(self.q_target_file_path, 'r') as q_target_file:
        #     q_target_str = q_target_file.readline()
        #     q_target_str = q_target_str.replace('[', '').replace(']', '').replace(',', '')
        #     q_target_str = q_target_str.split(' ')[0:-1]
        #     q_target = list(map(eval, q_target_str))
        #
        # return loss, scores, time_step_when_episode_end, reward_every_time_step, q_target
        with open(self.score_every_episode_file_path, 'r') as score_every_episode_file:
            scores_str = score_every_episode_file.readline().split(" ")
            sum = 0
            for i in scores_str:
                print(i)
                sum = sum + int(i)
            scores_str = scores_str[0:-1]
            scores = list(map(eval, scores_str))
            return sum
    def get_weight(self, shape):
        # w = tf.truncated_normal(shape, stddev=0.01)
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.01))
        return w

    # 定义偏置b 神经网络模型中使用

    def get_bias(self, shape):
        b = tf.Variable(tf.constant(0.01, shape=shape))
        return b

    # 定义卷积操作 步长为[1,1,1,1],边界填充
    # padding: 全零填充 'SAME'表示使用
    # tf.nn.conv2d(输入描述，卷积核描述，核滑动步长，padding)
    # eg. yf.nn.conv2d([BATCH(一次喂入多少图片),5,5(分辨率),1(通道数)]，[3,3(行列分辨率),1(通道数)],[3,3(行列分辨率),1(通道数),16(核数),
    # [1,1,1,1],padding='VALID')

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    # 定义最大池化操作
    # tf.nn.max_pool(输入描述，池化核描述(仅大小)，池化核滑动步长，padding)

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (80, 80, 1))


def playFlappyBird():
    # # Step 1: init BrainDQN
    # actions = 2
    # brain = BrainDQN(actions)
    # # Step 2: init Flappy Bird Game
    # flappyBird = game.GameState()
    # # Step 3: play game
    # # Step 3.1: obtain init state
    # action0 = np.array([1, 0])  # do nothing
    # observation0, reward0, terminal, _ = flappyBird.frame_step(action0)
    # observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    # ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    # brain.setInitState(observation0)
    #
    # # Step 3.2: run the game
    # while 1 != 0:
    #     action = brain.getAction()
    #     nextObservation, reward, terminal, score = flappyBird.frame_step(action)
    #     nextObservation = preprocess(nextObservation)
    #     brain.setPerception(nextObservation, action, reward, terminal, score)
    actions = 2
    brain = BrainDQN(actions)

def main():
    playFlappyBird()


if __name__ == '__main__':
    main()
