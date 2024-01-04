import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import hiddenlayer as h
import torch.nn.functional as F
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

import utils
from utils import (construct_model, generate_data)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configuration file')
parser.add_argument('--test', action="store_true", help='test program')
parser.add_argument("--plot", action="store_true", help="plot network graph")
parser.add_argument("--save", action="store_true", help="save model")
args = parser.parse_args()

config_filename = args.config

with open(config_filename, 'r') as f:
    config = json.loads(f.read())

print(json.dumps(config, sort_keys=True, indent=4))

net = construct_model(config)

batch_size = config['batch_size']
num_of_vertices = config['num_of_vertices']
graph_signal_matrix_filename = config['graph_signal_matrix_filename']
if isinstance(config['ctx'], list):
    print("sorry, I do not wirte about distributed, so you only can use one GPU to train")
    pass
elif isinstance(config['ctx'], int):
    ctx = torch.device("cuda:" + str(config['ctx']))

loaders = []
true_values = []
for idx, (x, y) in enumerate(generate_data(graph_signal_matrix_filename)):
    if args.test:
        x = x[: 100]
        y = y[: 100]
    y = y.squeeze(axis=-1)
    print(x.shape, y.shape)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    loaders.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y),
                                               batch_size=batch_size, shuffle=True))
    training_samples = x.shape[0]

train_loader, val_loader, test_loader = loaders

global_epoch = 1
global_train_steps = training_samples // batch_size + 1
all_info = []
epochs = config['epochs']

mod = net.to(ctx)

# The parameters of the mod
num_of_parameters = 0
for param in mod.parameters():
    num_of_parameters += param.numel()
print("Number of Parameters: {}".format(num_of_parameters), flush=True)

for p in net.parameters():
    if p.dim() > 1:
        nn.init.xavier_normal_(p, gain=0.0003)
    else:
        nn.init.uniform_(p)

if args.plot:
    # tensor, shape is (B, T, N, C)
    x = torch.randn(32, 12, 358, 1).to(ctx)
    # tensor, shape is (B, T, N)
    y = torch.randn(32, 12, 358).to(ctx)
    dot = make_dot(mod(x, y), params=dict(mod.named_parameters()))
    dot.render('model', format='png')


def training(epochs, config):

    global global_epoch, best_params
    writer = SummaryWriter('./log/PEMS03', flush_secs=20)
    lowest_val_loss = 1e6
    criterion = torch.nn.SmoothL1Loss()
    base_lr = config['learning_rate']

    optimizer = optim.Adam(net.parameters(), lr=base_lr, eps=1.0e-8, weight_decay=0, amsgrad=False)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=[15,40,70,105,145],
                                                        gamma=0.3)
    exp=1

    for epoch in range(epochs):
        t = time.time()
        info = [global_epoch]
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        net.train()

        for idx, data in enumerate(train_loader):
            x, y = data
            x, y = x.to(ctx), y.to(ctx)
            output = net(x)
            loss=criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

            tmae = utils.masked_mae(output, y ).item()
            tmape = utils.masked_mape(output, y , 0.0).item()
            trmse = utils.masked_rmse(output, y , 0.0).item()

            train_loss.append(loss.item())
            train_mae.append(tmae)
            train_mape.append(tmape)
            train_rmse.append(trmse)

            if idx % 200 == 0:
                print(f"Epoch: {idx}, Loss: {train_loss[-1]}, MAE: {train_mae[-1]}, "
                      f"MAPE: {train_mape[-1]}, RMSE: {train_rmse[-1]}")

        lr_scheduler.step()

        valid_mae = []
        valid_mape = []
        valid_rmse = []
        net.eval()
        for idx, data in enumerate(val_loader):
            x,y = data
            x,y = x.to(ctx), y.to(ctx)
            output = net(x)

            tmae = utils.masked_mae(output, y ).item()
            tmape = utils.masked_mape(output, y , 0.0).item()
            trmse = utils.masked_rmse(output, y , 0.0).item()

            valid_mae.append(tmae)
            valid_mape.append(tmape)
            valid_rmse.append(trmse)

        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        print(f"Epoch: {epoch}, Train Loss: {mtrain_loss}, Train MAE: {mtrain_mae}, "
              f"Train MAPE: {mtrain_mape}, Train RMSE: {mtrain_rmse}, \n"
              f"Valid MAE: {mvalid_loss}, "
              f"Valid MAPE: {mvalid_mape}, Valid RMSE: {mvalid_rmse}")

        writer.add_scalar('train/Loss', mtrain_loss, epoch)
        writer.add_scalar('train/MAE', mtrain_mae, epoch)
        writer.add_scalar('train/MAPE', mtrain_mape, epoch)
        writer.add_scalar('train/RMSE', mtrain_mape, epoch)

        writer.add_scalar('valid/MAE', mvalid_loss, epoch)
        writer.add_scalar('valid/MAPE', mvalid_mape, epoch)
        writer.add_scalar('valid/RMSE', mvalid_rmse, epoch)

        if mvalid_loss <= lowest_val_loss:
            lowest_val_loss = mvalid_loss
            best_params = model.state_dict()
            torch.save(net.state_dict(), 'save/exp_{}_{}.pth'.format(exp, str(round(lowest_val_loss, 2))))
            exp+=1

        global_epoch += 1
    #取最好的进行相关的测试
    net.load_state_dict(best_params)
    test_mae = []
    test_mape = []
    test_rmse = []
    net.eval()
    for idx, data in enumerate(test_loader):
        x,y = data
        x,y = x.to(ctx), y.to(ctx)
        output = net(x)

        tmae = utils.masked_mae(output, y ).item()
        tmape = utils.masked_mape(output, y , 0.0).item()
        trmse = utils.masked_rmse(output, y , 0.0).item()

        test_mae.append(tmae)
        test_mape.append(tmape)
        test_rmse.append(trmse)

    mtest_mae = np.mean(test_mae)
    mtest_mape = np.mean(test_mape)
    mtest_rmse = np.mean(test_rmse)

    print(f"The result of the Test is:\n"
          f"Test MAE: {mtest_mae}, "
          f"Test MAPE: {mtest_mape}, Test RMSE: {mtest_rmse}")


if args.test:
    epochs = 5

training(epochs, config)
