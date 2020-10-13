from config import opt
from data_handler import *
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from models import ImgModule, TxtModule, ImgMemory, TxtMemory
from utils import calc_map_k
import time
import os


def train(**kwargs):
    opt.parse(kwargs)

    pretrain_img_model = load_pretrain_model(opt.pretrain_model_path)

    train_L, query_L, retrieval_L, train_x, query_x, retrieval_x, train_y, query_y, retrieval_y = load_data(
        opt.data_path, opt.data_set)

    train_L = torch.from_numpy(train_L)
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    query_L = torch.from_numpy(query_L)
    query_x = torch.from_numpy(query_x)
    query_y = torch.from_numpy(query_y)

    retrieval_L = torch.from_numpy(retrieval_L)
    retrieval_x = torch.from_numpy(retrieval_x)
    retrieval_y = torch.from_numpy(retrieval_y)

    y_dim = query_y.shape[1]
    num_class = query_L.shape[1]

    print('...loading and splitting data finish')

    img_model = ImgModule(opt.bit, pretrain_img_model)
    txt_model = TxtModule(y_dim, opt.bit)
    img_memory_net = ImgMemory(num_class, opt.bit)
    txt_memory_net = TxtMemory(num_class, opt.bit)

    txt_model.load(opt.load_txt_path, use_gpu=True)

    if opt.use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()
        img_memory_net = img_memory_net.cuda()
        txt_memory_net = txt_memory_net.cuda()

    num_train = train_y.shape[0]

    F_buffer = torch.randn(num_train, opt.bit)
    G_buffer = torch.randn(num_train, opt.bit)

    if opt.use_gpu:
        train_L = train_L.cuda()
        F_buffer = F_buffer.cuda()
        G_buffer = G_buffer.cuda()

    Sim = calc_neighbor(train_L, train_L)
    B = torch.sign(F_buffer + G_buffer)
    batch_size = opt.batch_size

    # F_buffer = img_model()
    for i in tqdm(range(num_train // batch_size)):
        index = range(num_train)
        ind = index[i * batch_size: ((i + 1) * batch_size if (i + 1) * batch_size < num_train else num_train)]

        image = Variable(train_x[ind].type(torch.float))
        if opt.use_gpu:
            image = image.cuda()

        cur_f = img_model(image)  # cur_f: (batch_size, bit)
        F_buffer[ind, :] = cur_f.data

    # G_buffer = txt_model()
    for i in tqdm(range(num_train // batch_size)):
        index = range(num_train)
        ind = index[i * batch_size: ((i + 1) * batch_size if (i + 1) * batch_size < num_train else num_train)]

        text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
        text = Variable(text)
        if opt.use_gpu:
            text = text.cuda()

        cur_g = txt_model(text)  # cur_f: (batch_size, bit)
        G_buffer[ind, :] = cur_g.data


    img_centroids = calc_centroids(F_buffer, train_L, num_train, opt.bit)
    text_centroids = calc_centroids(G_buffer, train_L, num_train, opt.bit)

    F_buffer = torch.randn(num_train, opt.bit)
    G_buffer = torch.randn(num_train, opt.bit)

    if opt.use_gpu:
        train_L = train_L.cuda()
        F_buffer = F_buffer.cuda()
        G_buffer = G_buffer.cuda()

    lr_base = opt.base_lr
    lr_memory = opt.memory_lr
    memory_optimizer_img = SGD(img_memory_net.parameters(), lr=lr_memory)
    memory_optimizer_txt = SGD(txt_memory_net.parameters(), lr=lr_memory)
    optimizer_img = SGD(img_model.parameters(), lr=lr_base)
    optimizer_txt = SGD(txt_model.parameters(), lr=lr_base)

    learning_rate = np.linspace(opt.base_lr, np.power(10, -6.), opt.max_epoch + 1)
    memory_learning_rate = np.linspace(opt.memory_lr, np.power(10, -6.), opt.max_epoch + 1)
    result = {
        'loss': []
    }

    ones = torch.ones(batch_size, 1)
    ones_ = torch.ones(num_train - batch_size, 1)
    unupdated_size = num_train - batch_size

    max_mapi2t = max_mapt2i = 0.

    for epoch in range(opt.max_epoch):
        # train image net
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(train_L[ind, :])
            image = Variable(train_x[ind].type(torch.float))
            if opt.use_gpu:
                image = image.cuda()
                sample_L = sample_L.cuda()
                ones = ones.cuda()
                ones_ = ones_.cuda()

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)

            cur_f = img_model(image)  # cur_f: (batch_size, bit)
            cur_f = img_memory_net(cur_f, img_centroids)

            F_buffer[ind, :] = cur_f.data
            F = Variable(F_buffer)
            G = Variable(G_buffer)

            theta_x = 1.0 / 2 * torch.matmul(cur_f, G.t())
            logloss_x = -torch.sum(S * theta_x - torch.log(1.0 + torch.exp(theta_x)))
            quantization_x = torch.sum(torch.pow(B[ind, :] - cur_f, 2))
            balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F[unupdated_ind].t().mm(ones_), 2))
            loss_x = logloss_x + opt.gamma * quantization_x + opt.eta * balance_x
            loss_x /= (batch_size * num_train)

            optimizer_img.zero_grad()
            memory_optimizer_img.zero_grad()
            loss_x.backward()
            optimizer_img.step()
            memory_optimizer_img.step()

        # train txt net
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(train_L[ind, :])
            text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            text = Variable(text)
            if opt.use_gpu:
                text = text.cuda()
                sample_L = sample_L.cuda()

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)
            cur_g = txt_model(text)  # cur_f: (batch_size, bit)
            cur_g = txt_memory_net(cur_g, text_centroids)
            G_buffer[ind, :] = cur_g.data
            F = Variable(F_buffer)
            G = Variable(G_buffer)

            # calculate loss
            # theta_y: (batch_size, num_train)
            theta_y = 1.0 / 2 * torch.matmul(cur_g, F.t())
            logloss_y = -torch.sum(S * theta_y - torch.log(1.0 + torch.exp(theta_y)))
            quantization_y = torch.sum(torch.pow(B[ind, :] - cur_g, 2))
            balance_y = torch.sum(torch.pow(cur_g.t().mm(ones) + G[unupdated_ind].t().mm(ones_), 2))
            loss_y = logloss_y + opt.gamma * quantization_y + opt.eta * balance_y
            loss_y /= (num_train * batch_size)

            optimizer_txt.zero_grad()
            memory_optimizer_txt.zero_grad()
            loss_y.backward()
            optimizer_txt.step()
            memory_optimizer_txt.step()

        # update B
        B = torch.sign(F_buffer + G_buffer)

        # calculate total loss
        loss = calc_loss(B, F, G, Variable(Sim), opt.gamma, opt.eta)

        print('...epoch: %3d, loss: %3.3f, base_lr: %f' % (epoch + 1, loss.data, lr_base))
        result['loss'].append(float(loss.data))

        # if opt.valid:
        #     mapi2t, mapt2i = valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L)
        #     print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))
        #     if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
        #         max_mapi2t = mapi2t
        #         max_mapt2i = mapt2i
        #         img_model.save(img_model.module_name + '.pth')
        #         txt_model.save(txt_model.module_name + '.pth')

        lr_base = learning_rate[epoch + 1]
        lr_memory = memory_learning_rate[epoch + 1]

        img_centroids = calc_centroids(F_buffer, train_L, num_train, opt.bit)
        text_centroids = calc_centroids(G_buffer, train_L, num_train, opt.bit)

        # set learning rate
        for param in optimizer_img.param_groups:
            param['lr'] = lr_base
        for param in optimizer_txt.param_groups:
            param['lr'] = lr_base
        for param in memory_optimizer_img.param_groups:
            param['lr'] = lr_memory
        for param in memory_optimizer_txt.param_groups:
            param['lr'] = lr_memory

    print('...training procedure finish')
    if opt.valid:
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
        result['mapi2t'] = max_mapi2t
        result['mapt2i'] = max_mapt2i
    else:
        mapi2t, mapt2i = valid(img_model, txt_model, img_memory_net, txt_memory_net, img_centroids, text_centroids, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L, save=True)

        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
        result['mapi2t'] = mapi2t
        result['mapt2i'] = mapt2i

    write_result(result)


def valid(img_model, txt_model, img_memory_net, text_memory_net, img_centroids, text_centoids, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L, save=False):
    qBX = generate_image_code(img_model, img_memory_net, query_x, img_centroids, opt.bit)
    qBY = generate_text_code(txt_model, text_memory_net, query_y, text_centoids, opt.bit)
    rBX = generate_image_code(img_model, img_memory_net, retrieval_x, img_centroids, opt.bit)
    rBY = generate_text_code(txt_model, text_memory_net, retrieval_y, text_centoids, opt.bit)

    dir_name = 'hashcodes' + time.strftime("%Y_%m_%d_%H%M", time.localtime())
    os.makedirs(dir_name)
    if save:
        np.save(os.path.join(dir_name, 'qBX.npy'), qBX.cpu().numpy())
        np.save(os.path.join(dir_name, 'qBY.npy'), qBY.cpu().numpy())
        np.save(os.path.join(dir_name, 'rBX.npy'), rBX.cpu().numpy())
        np.save(os.path.join(dir_name, 'rBY.npy'), rBY.cpu().numpy())
        print("Hash codes saved.\n")

    num_class = query_L.shape[1]
    num_test = query_L.shape[0]
    index = np.arange(num_test)
    mlist_i2t = []
    mlist_t2i = []

    for i in range(num_class):
        ind = index[query_L[:, i] == 1]
        class_mapi2t = calc_map_k(qBX[ind, :], rBY, query_L[ind, :], retrieval_L)
        class_mapt2i = calc_map_k(qBY[ind, :], rBX, query_L[ind, :], retrieval_L)
        mlist_i2t.append(class_mapi2t)
        mlist_t2i.append(class_mapt2i)

    f = open(os.path.join(dir_name, 'maplist.txt'), "w")
    f.write("i2t: " + str(mlist_i2t))
    f.write("\n")
    f.write("t2i: " + str(mlist_t2i))

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    return mapi2t, mapt2i


def test(**kwargs):
    opt.parse(kwargs)

    train_L, query_L, retrieval_L, train_x, query_x, retrieval_x, train_y, query_y, retrieval_y = load_data(
        opt.data_path)

    y_dim = query_y.shape[1]

    print('...loading and splitting data finish')

    img_model = ImgModule(opt.bit)
    txt_model = TxtModule(y_dim, opt.bit)

    if opt.load_img_path:
        img_model.load(opt.load_img_path)

    if opt.load_txt_path:
        txt_model.load(opt.load_txt_path)

    if opt.use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()

    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    if opt.use_gpu:
        query_L = query_L.cuda()
        retrieval_L = retrieval_L.cuda()

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    print('...test MAP: MAP(i->t): %3.3f, MAP(t->i): %3.3f' % (mapi2t, mapt2i))


def calc_neighbor(label1, label2):
    # calculate the similar matrix
    label1 = label1.type(torch.cuda.FloatTensor)
    label2 = label2.type(torch.cuda.FloatTensor)
    if opt.use_gpu:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    else:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    return Sim


def calc_loss(B, F, G, Sim, gamma, eta):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
    loss = term1 + gamma * term2 + eta * term3
    print('term1%3.3f, term2%3.3f, term3%3.3f' % (term1, term2, term3))
    return loss


def generate_image_code(img_model, img_memory_net, X, centroids, bit):
    batch_size = opt.batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = X[ind].type(torch.float)
        if opt.use_gpu:
            image = image.cuda()
        cur_f = img_model(image)
        cur_f = img_memory_net(cur_f, centroids)
        B[ind, :] = cur_f.data
    B = torch.sign(B)
    return B


def generate_text_code(txt_model, txt_memory_net, Y, centroids, bit):
    batch_size = opt.batch_size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
        if opt.use_gpu:
            text = text.cuda()
        cur_g = txt_model(text)
        cur_g = txt_memory_net(cur_g, centroids)
        B[ind, :] = cur_g.data
    B = torch.sign(B)
    return B


def write_result(result):
    import os
    with open(os.path.join(opt.result_dir, 'result.txt'), 'w') as f:
        for k, v in result.items():
            f.write(k + ' ' + str(v) + '\n')

def calc_centroids(feature, label, num_train, bit):
    num_class = label.shape[1]
    centroids = torch.zeros((num_class, bit)).cuda()
    for j in range(num_class):
        n = 0
        for i in range(num_train):
            if label[i, j] == 1:
                centroids[j, :] = centroids[j, :] + feature[i, :]
                n += 1
        centroids[j, :] /= n
    return centroids


def help():
    """
    打印帮助的信息： python file.py help
    """
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --lr=0.01
            python {0} help
    avaiable args:'''.format(__file__))
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'):
            print('\t\t{0}: {1}'.format(k, v))


if __name__ == '__main__':
    torch.cuda.set_device(1)
    train()
