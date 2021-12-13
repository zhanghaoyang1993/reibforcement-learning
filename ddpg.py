
import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
lr_mu = 0.0005  # learning rate of actor network，（25）in the paper 0.0001
lr_q = 0.001  # learning rate of critic network，（24）in the paper 0.001
gamma = 0.99  # discount factor， 0.7 in the paper
batch_size = 32
buffer_limit = 50000
tau = 0.005  # for target soft update （26） （27） in the paper


class ReplayBuffer:  # 对buffer进行操作，返回s_lst（32*3）, a_lst（32*1）, r_lst（32*1）, s_prime_lst（32*3）, done_mask_lst（32*1）的tensor,  done_mask_lst代表是否为最后一步
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)  # 创建一个固定长度的队列。当有新纪录加入而队列已满时会自动移除最老的那条记录

    def put(self, transition):
        self.buffer.append(transition)  # 将transition加到buffer中

    def sample(self, n): # sample n randomly for updating the network
        mini_batch = random.sample(self.buffer, n)  # sample from buffer
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []  # done_mask_lst代表是否为最后一步

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)  # s为1*3的array，s_lst为32*3的list，类型float
            a_lst.append([a])  # a为1*1，a_lst为32*1的list，类型为float
            r_lst.append([r])  # r为1*1，r_lst为32*1的list，类型为float
            s_prime_lst.append(s_prime)  # s_prime为3*1的array，s_prime_lst为32*3的list，类型float
            done_mask_lst.append([done_mask])  # done_mask为1*1的array，done_mask_lst为32*1的list，类型bool

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), \
               torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(done_mask_lst)  # 将array转化为tensor

    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):  # 搭建actor的神经网络，输入state输出action
    def __init__(self):
        super(MuNet, self).__init__()  # 3*128*64*1
        self.fc1 = nn.Linear(3, 128)  # state是3维
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)  # action是1维

    def forward(self, x):
        x = F.relu(self.fc1(x))  #使用relu激活函数
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * 2  # 乘以2是因为tanh的输出在-1,1之间，该环境的action space在-2,2之间

        return mu  # 输出action


class QNet(nn.Module):  # 搭建critic神经网络，输入state和action，输出Q值
    def __init__(self):
        super(QNet, self).__init__()  # 4*128*32*1

        self.fc_s = nn.Linear(3, 64)  # state是3维
        self.fc_a = nn.Linear(1, 64)  # action是1维
        self.fc_q = nn.Linear(128, 32)
        self.fc_3 = nn.Linear(32, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)  # 将h1,h2合并为[h1, h2]
        q = F.relu(self.fc_q(cat))
        q = self.fc_3(q)  # 未设置输出的激活函数，因为q值范围未知

        return q


#  用于explore的自相关OU噪声,给μ加噪音   dxt = -theta*(xt-mu)*dt+sigma*dWt，要根据论文搞成高斯白噪音
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.1, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)  # 构造一个和self.mu维度一致的全0矩阵

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x


def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):  # 更新q_optimizer和mu_optimizer的参数，用于mu网络和q网络

    s, a, r, s_prime, done_mask = memory.sample(batch_size)  # 从memory中采样batch_size个训练样本，s：32*3

    target = r + gamma * q_target(s_prime, mu_target(s_prime))  # 通过Q_target网络计算r+gamma*Q'(s_(n+1),a_(n+1))的值
    q_loss = F.smooth_l1_loss(q(s, a), target.detach())  # 计算TD error，Q用smoothl1loss作为损失函数
    q_optimizer.zero_grad()  # 将网络中的参数设为0
    q_loss.backward()  # 反向传播
    q_optimizer.step()  # 更新网络参数

    mu_loss = -q(s, mu(s)).mean()
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


def soft_update(net, net_target):  # 更新target网络的参数
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def main():
    env = gym.make('Pendulum-v0')  # environment
    memory = ReplayBuffer()

    q, q_target = QNet(), QNet()  # 创建q，q_target网络
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0  # 初始化score
    print_interval = 20

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))  # 初始的μ为np.zeros(1)，即1维

    for n_epi in range(10000):
        s = env.reset()  # 重置开始状态,在任意状态出发
        for t in range(300):  # maximum length of episode is 300 for Pendulum-v0
            a = mu(torch.from_numpy(s).float())  # action: μ(s),from_numpy(s)将s从numpy转化为torch
            a = a.item() + ou_noise()[0]  # action加niose
            s_prime, r, done, info = env.step([a])  # 选择动作a后返回r，下一状态s和是否最终状态的信息
            memory.put((s, a, r / 100.0, s_prime, done))  # 储存近memory中
            score += r  # 将每个时刻t的r累加起来
            s = s_prime  # 更新s

            if done:
                break

        # 当memory size大于2000时，buffer足够大，开始更新参数
        if memory.size() > 2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q, q_target)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()
