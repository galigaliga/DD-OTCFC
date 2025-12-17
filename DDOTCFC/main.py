import argparse
import math
import time
import random
import torch
import torch.nn as nn
from test_model.DDOTCFC import Model
import importlib
from util import *
from trainer import Optim
import pandas as pd
from metrics import *
from ptflops import get_model_complexity_info
import os
from get_config import get_config
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
import numpy as np
warnings.filterwarnings("ignore")

test_model_name = ('DDOTCFC')
config = get_config(test_model_name)
pred_length = config.pred_len
seq_len = config.seq_len

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        # [64,12,3]
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

    scale = data.scale.expand(predict.size(0), predict.size(1), 3).cpu().numpy()

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    mape = MAPE(predict * scale, Ytest * scale)
    mae = MAE(predict * scale, Ytest * scale)
    rmse = RMSE(predict * scale, Ytest * scale)

    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return mae, mape, correlation, rmse


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    total_mae_loss = 0

    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        tx = X
        ty = Y
        tx = torch.squeeze(tx).transpose(1, 2)

        output = model(tx)
        output = torch.squeeze(output)
        scale = data.scale.expand(output.size(0), output.size(1), 3)

        loss = mape_loss(ty * scale, output * scale)
        loss_mae = MAE((ty * scale).cpu().detach().numpy(), (output * scale).cpu().detach().numpy())

        loss_mse = MSE(ty * scale, output * scale)
        loss.backward()
        total_loss += loss.item()
        total_mae_loss += loss_mae.item()
        grad_norm = optim.step()
        iter += 1

    return total_loss / iter, total_mae_loss / iter


def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:

        _dict = {}
        for _, param in enumerate(model.named_parameters()):
            total_params = param[1].numel()
            k = param[0].split('.')[0]
            if k in _dict.keys():
                _dict[k] += total_params
            else:
                _dict[k] = 0
                _dict[k] += total_params
        total_param = sum(p.numel() for p in model.parameters())
        bytes_per_param = 1
        total_bytes = total_param * bytes_per_param
        total_megabytes = total_bytes / (1024 * 1024)
        return total_param, total_megabytes, _dict


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/dataset_input_jiuzheng.csv',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--seq_in_len', type=int, default=config.seq_len, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=config.pred_len, help='output sequence length')
parser.add_argument('--horizon', type=int, default=config.pred_len)
parser.add_argument('--layers', type=int, default=5, help='number of layers')
parser.add_argument('--batch_size', type=int, default=config.batchsize, help='batch size')
parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--epochs', type=int, default=config.epochs, help='')

args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    seed = 2022
    fix_seed(seed)
    test_mae_history = []

    fin = open(args.data)
    rawdat = np.loadtxt(fin, delimiter=',', skiprows=1)
    print(rawdat.shape)

    Data = DataLoaderS(args.data, 0.8, 0.1, device, args.horizon, args.seq_in_len, args.normalize)

    model = Model(config)
    model = model.to(device)

    flops, params = get_model_complexity_info(model, (1, 12, config.seq_len), as_strings=True, print_per_layer_stat=False)
    print('flops: ', flops, 'params: ', params)
    print('------------------------------------------------------')

    total_param, total_megabytes, _dict = count_parameters(model)
    print("Total megabytes:", total_megabytes, "M")
    print("Total parameters:", total_param)
    print(args)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)

    model = Model(config).to(device)

    optim = Optim(
        model.parameters(), args.optim, config.lr, args.clip, 'min', config.weightdecay, config.decaypatience,
        lr_decay=args.weight_decay
    )
    best_test_mae = float('inf')
    best_epoch_metrics = {}
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            train_loss, train_mae_loss = train(Data, Data.train[0], Data.train[1][:, :, :3], model, criterion, optim,
                                               args.batch_size)
            val_mae, val_mape, val_corr, val_rmse = evaluate(Data, Data.valid[0], Data.valid[1][:, :, :3], model,
                                                             evaluateL2,
                                                             evaluateL1,
                                                             args.batch_size)
            optim.lronplateau(val_mape)

            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_mape_loss {:5.4f}| train_mae_loss {:5.4f} | valid mae {:5.4f} | valid mape {:5.4f} | valid corr  {:5.4f} | learning rate  {:f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss,train_mae_loss, val_mae, val_mape, val_corr, optim.optimizer.param_groups[0]['lr']), flush=True)


            if epoch % 1 == 0:
                test_mae, test_mape, test_corr, test_rmse = evaluate(Data, Data.test[0], Data.test[1][:, :, :3], model,
                                                                     evaluateL2,
                                                                     evaluateL1,
                                                                     args.batch_size)
                test_mae_history.append(test_mae)
                print("test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f} | test rmse {:5.4f}".format(test_mae,
                                                                                                            test_mape,
                                                                                                            test_corr,
                                                                                                            test_rmse),
                      flush=True)

                if test_mae < best_test_mae:
                    best_test_mae = test_mae
                    best_epoch_metrics = {
                        'epoch': epoch,
                        'test_mae': test_mae,
                        'test_mape': test_mape,
                        'test_corr': test_corr,
                        'test_rmse': test_rmse
                    }
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    if best_epoch_metrics:
        print(f"Best epoch: {best_epoch_metrics['epoch']}")
        print("Best epoch test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f} | test rmse {:5.4f}".format(
            best_epoch_metrics['test_mae'],
            best_epoch_metrics['test_mape'],
            best_epoch_metrics['test_corr'],
            best_epoch_metrics['test_rmse']
        ))
    print('-' * 89)

    if best_epoch_metrics:
        return best_epoch_metrics['test_mae'], best_epoch_metrics['test_mape'], best_epoch_metrics['test_corr']
    else:
        return None, None, None


if __name__ == "__main__":
    main()

