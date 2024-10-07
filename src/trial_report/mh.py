import glob
import os
import time
import numpy as np
import pandas as pd
import joblib
import copy
import random
from scipy import signal
from bagpy import bagreader
import json

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# from src.trial_report.utils import *
from utils import *


def butter_lowpass(cutoff, nyq_freq, order):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype="lowpass")

    return b, a


def butter_bandpass(lowcut, highcut, nyq_freq, order):
    low = float(lowcut) / nyq_freq
    high = float(highcut) / nyq_freq
    b, a = signal.butter(order, [low, high], btype="band")

    return b, a


def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order)
    y = signal.filtfilt(b, a, data)

    return y


def butter_bandpass_filter(data, low_freq, high_freq, nyq_freq, order):
    b, a = butter_bandpass(low_freq, high_freq, nyq_freq, order)
    y = signal.filtfilt(b, a, data)

    return y


def z_score_normalization_previous(data):
    normalized = []

    for value in data:
        normalized_num = (value - np.mean(data)) / np.std(data)
        normalized.append(normalized_num)

    return normalized


def z_score_normalization(time_data, vout_data, force_data, force_idx):
    time_data = pd.DataFrame(time_data)
    vout_data = pd.DataFrame(vout_data)
    force_data = pd.DataFrame(force_data)
    time_data.reset_index(drop=True, inplace=True)
    vout_data.reset_index(drop=True, inplace=True)
    force_data.reset_index(drop=True, inplace=True)

    for _ in np.arange(20):
        min_value = min(force_data.iloc[:, force_idx].values)
        np_data = np.array(force_data.iloc[:, force_idx])
        norm_index = np.where(np_data == min_value)[0]
        time_data = time_data.drop(norm_index)
        vout_data = vout_data.drop(norm_index)
        force_data = force_data.drop(norm_index)
        time_data.reset_index(drop=True, inplace=True)
        vout_data.reset_index(drop=True, inplace=True)
        force_data.reset_index(drop=True, inplace=True)
    for _ in np.arange(20):
        max_value = max(force_data.iloc[:, force_idx].values)
        np_data = np.array(force_data.iloc[:, force_idx])
        norm_index = np.where(np_data == max_value)[0]
        time_data = time_data.drop(norm_index)
        vout_data = vout_data.drop(norm_index)
        force_data = force_data.drop(norm_index)
        time_data.reset_index(drop=True, inplace=True)
        vout_data.reset_index(drop=True, inplace=True)
        force_data.reset_index(drop=True, inplace=True)

    # mean, std calculation
    mean_value = np.mean(force_data.iloc[:, force_idx])
    std_value = np.std(force_data.iloc[:, force_idx])

    np_data = np.array(force_data.iloc[:, force_idx])
    norm_index = np.where(np_data != mean_value)[0]
    force_data.iloc[:, force_idx] -= mean_value
    force_data.iloc[norm_index, force_idx] /= std_value

    fig = plt.figure()
    plt.plot(force_data.iloc[:, force_idx])
    plt.title(force_idx)
    # plt.close()

    return time_data, vout_data, force_data


def calculate_derivative(x, y):
    f_d = np.gradient(np.array(y), np.array(x))
    f_dd = np.gradient(f_d, np.array(x))

    return f_d, f_dd


def min_max_normalization(time_data, vout_data, force_data, force_idx):
    time_data = pd.DataFrame(time_data)
    vout_data = pd.DataFrame(vout_data)
    force_data = pd.DataFrame(force_data)
    time_data.reset_index(drop=True, inplace=True)
    vout_data.reset_index(drop=True, inplace=True)
    force_data.reset_index(drop=True, inplace=True)

    for _ in np.arange(20):
        min_value = min(force_data.iloc[:, force_idx].values)
        np_data = np.array(force_data.iloc[:, force_idx])
        norm_index = np.where(np_data == min_value)[0]
        time_data = time_data.drop(norm_index)
        vout_data = vout_data.drop(norm_index)
        force_data = force_data.drop(norm_index)
        time_data.reset_index(drop=True, inplace=True)
        vout_data.reset_index(drop=True, inplace=True)
        force_data.reset_index(drop=True, inplace=True)
    for _ in np.arange(20):
        max_value = max(force_data.iloc[:, force_idx].values)
        np_data = np.array(force_data.iloc[:, force_idx])
        norm_index = np.where(np_data == max_value)[0]
        time_data = time_data.drop(norm_index)
        vout_data = vout_data.drop(norm_index)
        force_data = force_data.drop(norm_index)
        time_data.reset_index(drop=True, inplace=True)
        vout_data.reset_index(drop=True, inplace=True)
        force_data.reset_index(drop=True, inplace=True)

    # rank option!!!!
    # min_rank = force_data.iloc[:, force_idx].rank(
    #     method='dense', ascending=True)
    # max_rank = force_data.iloc[:, force_idx].rank(
    #     method='dense', ascending=False)

    # min_idx = np.where(np.array(min_rank) == 100.0)[0][0]
    # max_idx = np.where(np.array(max_rank) == 100.0)[0][0]

    # print(min_idx)
    # print(max_idx)

    # min_value = force_data.iloc[min_idx, force_idx]
    # max_value = force_data.iloc[max_idx, force_idx]
    # print(min_value)
    # print(max_value)
    # interval = float(abs(max_value - min_value))

    # mean +- 2std option!!!!
    mean_value = np.mean(force_data.iloc[:, force_idx])
    std_value = np.std(force_data.iloc[:, force_idx])
    min_ref_value = mean_value - 2.0 * std_value
    max_ref_value = mean_value + 2.0 * std_value

    min_df = abs(force_data.iloc[:, force_idx] - min_ref_value)
    max_df = abs(force_data.iloc[:, force_idx] - max_ref_value)

    min_rank = min_df.rank(method='dense', ascending=True)
    max_rank = max_df.rank(method='dense', ascending=True)

    min_idx = np.where(np.array(min_rank) == 1.0)[0][0]
    max_idx = np.where(np.array(max_rank) == 1.0)[0][0]

    print(min_idx)
    print(max_idx)

    min_value = force_data.iloc[min_idx, force_idx]
    max_value = force_data.iloc[max_idx, force_idx]
    print(min_value)
    print(max_value)
    interval = float(abs(max_value - min_value))

    # np_data = np.array(force_data.iloc[:, force_idx])
    # # delete upper max
    # upper_index = np.where(np_data > max_value)[0]
    # force_data.iloc[upper_index, force_idx] = max_value
    # # delete lower min
    # lower_index = np.where(np_data < min_value)[0]
    # force_data.iloc[lower_index, force_idx] = min_value

    force_data.iloc[:, force_idx] -= min_value
    np_data = np.array(force_data.iloc[:, force_idx])
    norm_index = np.where(np_data != min_value)[0]
    force_data.iloc[norm_index, force_idx] /= interval

    fig = plt.figure()
    plt.plot(force_data.iloc[:, force_idx])
    plt.title(force_idx)
    # plt.close()

    return time_data, vout_data, force_data


# Function for folder path reading
def folder_path_name(path, option=None, char=None, T_F=None):

    folder_name_path = str(path)
    folder_path = folder_name_path + "*"

    file_list = glob.glob(folder_path)
    file_name_list = os.listdir(folder_name_path)

    if option == "start":
        exp_list = [file for (file, name) in zip(file_list, file_name_list)
                    if name.startswith(str(char)) == int(T_F)]
        exp_name_list = [name for (file, name) in
                         zip(file_list, file_name_list)
                         if name.startswith(str(char)) == int(T_F)]

    elif option == "end":
        exp_list = [file for (file, name) in zip(file_list, file_name_list)
                    if name.endswith(str(char)) == int(T_F)]
        exp_name_list = [name for (file, name)
                         in zip(file_list, file_name_list)
                         if name.endswith(str(char)) == int(T_F)]

    elif option == "include":
        exp_list = [file for (file, name) in zip(file_list, file_name_list)
                    if str(char) in name]
        exp_name_list = [name for (file, name)
                         in zip(file_list, file_name_list)
                         if str(char) in name]

    else:
        exp_list = [file for (file, name) in zip(file_list, file_name_list)]
        exp_name_list = [name for (file, name)
                         in zip(file_list, file_name_list)]

    exp_list = [file.replace('\\', '/') for file in exp_list]

    return exp_list, exp_name_list


# LSTM model structure for calibration
class LSTM_Calib(nn.Module):
    def __init__(self, input_size=1, hidden_size_1=15, hidden_size_2=60,
                 hidden_size_3=20, num_layers=2, drop_p=0.3):
        super(LSTM_Calib, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.num_layers = num_layers
        self.dropout = drop_p

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True,
            num_layers=num_layers,
            dropout=drop_p
        )
        self.linear1 = nn.Linear(in_features=self.hidden_size_1,
                                 out_features=self.hidden_size_2)
        self.linear2 = nn.Linear(in_features=self.hidden_size_2,
                                 out_features=self.hidden_size_3)
        self.linear3 = nn.Linear(in_features=self.hidden_size_3,
                                 out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size_1).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size_1).requires_grad_()

        outn, (hn, cn) = self.lstm(x, (h0, c0))
        out1 = self.linear1(outn[:, :, -1])
        out2 = self.linear2(out1)
        out = self.linear3(out2)

        return out


# LSTM model structure for GRF estimation
class LSTM_GRF(nn.Module):
    def __init__(self, input_size=6, hidden_size_1=15, hidden_size_2=60,
                 hidden_size_3=20, num_layers=2, drop_p=0.3):
        super(LSTM_GRF, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.num_layers = num_layers
        self.dropout = drop_p

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True,
            num_layers=num_layers,
            dropout=drop_p
        )
        self.linear1 = nn.Linear(in_features=self.hidden_size_1,
                                 out_features=self.hidden_size_2)
        self.linear2 = nn.Linear(in_features=self.hidden_size_2,
                                 out_features=self.hidden_size_3)
        self.linear3 = nn.Linear(in_features=self.hidden_size_3,
                                 out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size_1, device=x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size_1, device=x.device).requires_grad_()

        outn, (hn, cn) = self.lstm(x, (h0, c0))
        out1 = self.linear1(outn[:, :, -1])
        out2 = self.linear2(out1)
        out = self.linear3(out2)

        return out


##############################################################################
# basic setting
##############################################################################
# with scheduler
def train(device, model, iterator, scheduler, loss_fn, clip):

    model.train()
    epoch_loss = 0

    tqdm_bar = tqdm(enumerate(iterator))

    for i, batch in tqdm_bar:

        src = batch[0].to(device)
        trg = batch[1].to(device)

        scheduler.zero_grad()

        output = model(src, trg)
        loss = loss_fn(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        scheduler.step()
        epoch_loss += loss.item()
        tqdm_bar.set_description(
            "Train loss: {:.6f}".format(loss.item()))

    scheduler.update()

    return epoch_loss / len(iterator)


def evaluate(device, model, iterator, loss_fn):

    model.eval()
    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch[0].to(device)
            trg = batch[1].to(device)

            output = model(src, trg)

            loss = loss_fn(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


class ScheduleOptim():
    def __init__(self, optimizer, n_warmup_steps=10, decay_rate=0.9):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.decay = decay_rate
        self.n_steps = 0
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_lr = optimizer.param_groups[0]['lr']

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        self._optimizer.step()

    def get_lr(self):
        return self.current_lr

    def update(self):
        if self.n_steps < self.n_warmup_steps:
            lr = self.n_steps / self.n_warmup_steps * self.initial_lr
        elif self.n_steps == self.n_warmup_steps:
            lr = self.initial_lr
        else:
            lr = self.current_lr * self.decay

        self.current_lr = lr
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

        self.n_steps += 1


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class weighted_RMSE_Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(weighted_RMSE_Loss, self).__init__()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(
            (y * (yhat - y) ** 2).sum() / (y.sum() + self.eps) + self.eps
            )
        return loss


# Dimension: Batch * input length * 1
class forceDataset(Dataset):
    def __init__(self, input_length=20, forecast_length=1,
                 data_dir="D:/OneDrive/AFO/AFO_FSR_DATA/CHAR_240426_280PL/" +
                 "Calibration/280Left_1/force_conversion_test50.csv"):
        data = pd.read_csv(data_dir, delimiter=",")[["vout", "force"]]
        data["idx"] = data.index
        self.data = np.array(data[["idx", "vout", "force"]])
        self.input_length = input_length
        self.forecast_length = forecast_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2]

    def force(self):
        return self.data[:, 2]

    def x_transform(self, x, idx, col_idx):
        if (int(idx) + 1) < self.input_length:
            x = np.append(self.data[0, col_idx] *
                          np.ones((1, self.input_length - int(idx) - 1)),
                          self.data[0:int(idx) + 1, col_idx])
        else:
            x = self.data[int(idx) + 1 - self.input_length:int(idx) + 1,
                          col_idx]
        return x

    def y_transform(self, y, idx, col_idx):
        if (int(idx) + 1) < self.forecast_length:
            y = np.append(self.data[0, col_idx] *
                          np.ones((1, self.forecast_length - int(idx) - 1)),
                          self.data[0:int(idx) + 1, col_idx])
        else:
            y = self.data[int(idx) + 1 - self.forecast_length:int(idx) + 1,
                          col_idx]
        return y

    def collate_fn(self, dataset):
        batch_x, batch_y = [], []
        for (idx, x, y) in dataset:
            x = self.x_transform(x, idx, 1)
            x = torch.from_numpy(x)
            x = x.unsqueeze(1)
            y = self.y_transform(y, idx, 2)
            y = torch.from_numpy(y)
            y = y.unsqueeze(1)
            batch_x.append(x)
            batch_y.append(y)
        batch_x = torch.stack(batch_x, dim=0).float()
        batch_y = torch.stack(batch_y, dim=0).float()
        return batch_x, batch_y


# Dimension: Batch * input length * sensor_num(= 6)
class grfDataset(Dataset):

    def __init__(self, data, input_length=20, forecast_length=1):
        data = data.loc[:, ["f1", "f2", "f3", "f4", "f5", "f6", "GRF"]]
        data["idx"] = data.index
        self.data = np.array(data[
            ["idx", "f1", "f2", "f3", "f4", "f5", "f6", "GRF"]])
        self.input_length = input_length
        self.forecast_length = forecast_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def label(self):
        return self.data[:, -1]

    def transform(self, data):
        idx = data[0]
        # x
        if (int(idx) + 1) < self.input_length:
            for num in np.arange(6):
                if num == 0:
                    x = np.append(self.data[0, num+1] *
                                  np.ones(
                                      (1, self.input_length - int(idx) - 1)),
                                  self.data[0:int(idx) + 1, num+1])
                    x = x.reshape(self.input_length, 1)
                else:
                    x_temp = np.append(self.data[0, num+1] *
                                       np.ones(
                                           (1,
                                            self.input_length - int(idx) - 1)),
                                       self.data[0:int(idx) + 1, num+1])
                    x_temp = x_temp.reshape(self.input_length, 1)
                    x = np.append(x, x_temp, axis=1)
        else:
            x = self.data[int(idx) + 1 - self.input_length:int(idx) + 1, 1:-1]
        # label
        if (int(idx) + 1) < self.forecast_length:
            y = np.append(self.data[0, -1] *
                          np.ones((1, self.forecast_length - int(idx) - 1)),
                          self.data[0:int(idx) + 1, -1])
        else:
            y = self.data[int(idx) + 1 - self.forecast_length:int(idx) + 1, -1]
        return x, y

    def collate_fn(self, dataset):
        batch_x, batch_y = [], []
        for each_data in dataset:
            x, y = self.transform(each_data)
            # s(= 20), 6
            x = torch.from_numpy(x)
            # f, 1
            y = torch.from_numpy(y)
            y = y.unsqueeze(1)
            batch_x.append(x)
            batch_y.append(y)
        batch_x = torch.stack(batch_x, dim=0).float()
        batch_y = torch.stack(batch_y, dim=0).float()
        return batch_x, batch_y


##############################################################################
# force model
class forceEncoder(nn.Module):
    def __init__(self,
                 in_dim=1,  # input feature
                 hid_dim=128,
                 num_layers=3,
                 bidirectional=False,
                 dropout=0.3):
        super(forceEncoder, self).__init__()

        self.input_size = in_dim
        self.hidden_size = hid_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=self.bidirectional,
                            num_layers=self.num_layers,
                            dropout=self.dropout)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input, hidden, cell):
        '''
        Forward method of LSTM Encoder Module

        INPUT
        - input: (Batch, source_len, in_dim=1)
        - hidden: initialized hidden state, (Layers, Batch, hid_dim)
            -> zero로 initialize
        - cell: initialized cell state, (Layers, Batch, hid_dim)
            -> zero로 initialize

        OUTPUT
        - output: output features from last layer, (Batch, source_len, hid_dim)
        - hidden: final hidden state for each element, (Layers, Batch, hid_dim)
        - cell: final cell state for each element, (Layers, Batch, hid_dim)

        '''
        # print(f'Encoder Before:, input: {input.shape}, h: {hidden.shape}, c: {cell.shape}')
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        # print(f'Encoder Final:, output: {output.shape}, h: {hidden.shape}, c: {cell.shape}')

        return output, hidden, cell


class forceDecoder(nn.Module):

    def __init__(self,
                 out_dim=1,  # output feature
                 hid_dim=128,
                 num_layers=3,
                 bidirectional=False,
                 dropout=0.3):
        super(forceDecoder, self).__init__()

        self.output_size = out_dim
        self.hidden_size = hid_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.output_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=self.bidirectional,
                            num_layers=self.num_layers,
                            dropout=self.dropout)
        self.dropout = nn.Dropout(self.dropout)
        self.regressor = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, cell):
        '''
        INPUT
        - input: previous step output in decoder (Batch, 1)
        - hidden: previous hidden state (Layers, Batch, hid_dim)
        - cell: previous cell state (Layers, Batch, hid_dim)

        OUTPUT
        - output: (Batch, 1, hid_dim) -> (Batch, hid_dim) -> (Batch, out_dim=1)
        - hidden: (Layers, Batch, hid_dim)
        - cell: (Layers, Batch, hid_dim)
        '''
        input = input.unsqueeze(1)  # (Batch, 1, out_dim)

        # print(f'Decoder Before:, input: {input.shape}, h: {hidden.shape}, c: {cell.shape}')
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        # print(f'Decoder After:, output: {output.shape}, h: {hidden.shape}, c: {cell.shape}')
        output = self.regressor(output.squeeze(1))  # (Batch, out_dim=1)
        # print(f'Decoder Final:, output: {output.shape}, h: {hidden.shape}, c: {cell.shape}')
        return output, hidden, cell


class forceSeq2Seq(nn.Module):
    # source_len=20, forecast_len=20 <- from input data
    def __init__(self,
                 device,
                 in_dim=1,
                 hid_dim=128,
                 out_dim=1,
                 num_layers=3,
                 bidirectional=False,
                 dropout=0.3):
        super(forceSeq2Seq, self).__init__()

        self.input_size = in_dim
        self.hidden_size = hid_dim
        self.output_size = out_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.device = device
        self.encoder = forceEncoder(self.input_size,
                                    self.hidden_size,
                                    self.num_layers,
                                    self.bidirectional,
                                    self.dropout)
        self.decoder = forceDecoder(self.output_size,
                                    self.hidden_size,
                                    self.num_layers,
                                    self.bidirectional,
                                    self.dropout)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        '''
        Forward method of LSTM Seq2Seq Module

        INPUT
        - src: source batched data (Batch, source_len, in_dim=1)
        - trg: target batched data (Batch, forecast_len, out_dim=1)

        OUTPUT
        - outputs of prediction (Batch, forecast_len, out_dim=1)
        '''
        batch_size, forecast_len, _ = trg.shape
        # Encoder (start from zero-hidden & zero-cell states)
        hidden_0 = torch.zeros(self.num_layers,
                               batch_size,
                               self.hidden_size).to(self.device)
        cell_0 = torch.zeros(self.num_layers,
                             batch_size,
                             self.hidden_size).to(self.device)
        output, hidden, cell = self.encoder(src, hidden_0, cell_0)

        # Decoder
        decoder_hidden = hidden
        decoder_cell = cell
        decoder_outputs = torch.zeros(forecast_len,
                                      batch_size,
                                      self.output_size).to(self.device)
        # decoder_input = trg[:, 0, :]  # (Batch, 1)
        # (Batch, 1)
        decoder_input = torch.zeros(batch_size,
                                    self.output_size).to(self.device)
        for t in range(forecast_len):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(
                decoder_input, decoder_hidden, decoder_cell)
            decoder_outputs[t] = decoder_output

            teacher_force = random.random() < teacher_forcing_ratio
            # input update (Batch, 1)
            if t < forecast_len - 1:
                decoder_input = trg[:, t+1, :] \
                    if teacher_force else decoder_output
            else:
                continue

        return decoder_outputs.permute(1, 0, 2)

    def predict(self, input, forecast_len=1):
        '''
        Predict method of LSTM Seq2Seq Module

        INPUT
        - input: (Batch, source_len, in_dim=1)
        - forecast_len: (default=1)

        OUTPUT
        - outputs of prediction (Batch, forecast_len, out_dim=1)
        '''
        batch_size, _, _ = input.shape
        # Encoder
        # Encoder (start from zero-hidden & zero-cell states)
        hidden_0 = torch.zeros(self.num_layers,
                               batch_size,
                               self.hidden_size).to(self.device)
        cell_0 = torch.zeros(self.num_layers,
                             batch_size,
                             self.hidden_size).to(self.device)
        output, hidden, cell = self.encoder(input, hidden_0, cell_0)

        # Decoder
        decoder_hidden = hidden
        decoder_cell = cell
        decoder_outputs = torch.zeros(forecast_len,
                                      batch_size,
                                      self.output_size).to(self.device)
        # decoder_input = input[:, -1, :]  # (Batch, 1)
        # (Batch, 1)
        decoder_input = torch.zeros(batch_size,
                                    self.output_size).to(self.device)
        for t in range(forecast_len):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(
                decoder_input, decoder_hidden, decoder_cell)
            decoder_outputs[t] = decoder_output

            # input update (Batch, 1)
            if t < forecast_len - 1:
                decoder_input = decoder_output
            else:
                continue

        return decoder_outputs.permute(1, 0, 2)


##############################################################################
# GRF model
class grfEncoder(nn.Module):
    def __init__(self,
                 in_dim=6,  # input feature
                 hid_dim=400,
                 num_layers=3,
                 bidirectional=True,
                 dropout=0.3):
        super(grfEncoder, self).__init__()

        self.input_size = in_dim
        self.hidden_size = int(hid_dim/2)
        self.dec_hidden_size = hid_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=self.bidirectional,
                            num_layers=self.num_layers,
                            dropout=self.dropout)
        self.fc = nn.Linear(2 * self.hidden_size, self.dec_hidden_size)
        self.act = nn.Tanh()

    def forward(self, input, hidden, cell):
        '''
        Forward method of LSTM Encoder Module (Bi-directional)

        INPUT
        - input: (Batch, source_len, in_dim=6)
        - hidden: initialized hidden state, (2*Layers, Batch, hid_dim/2)
            -> zero로 initialize
        - cell: initialized cell state, (2*Layers, Batch, hid_dim/2)
            -> zero로 initialize

        OUTPUT
        - output: output features from last layer,
            (Batch, source_len, dec_hid_dim)
        - hidden: final hidden state for each element,
            (Layers, Batch, dec_hid_dim)
        - cell: final cell state for each element,
            (Layers, Batch, dec_hid_dim)

        '''
        batch_size, _, _ = input.shape
        # print(f'Encoder Before:, input: {input.shape}, h: {hidden.shape}, c: {cell.shape}')
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        # print(f'Encoder Final:, output: {output.shape}, h: {hidden.shape}, c: {cell.shape}')
        final_hidden = hidden.view(
            self.num_layers, 2, batch_size, self.hidden_size)
        final_cell = cell.view(
            self.num_layers, 2, batch_size, self.hidden_size)
        # print(f'Encoder After view:, h: {final_hidden.shape}, c: {final_cell.shape}')
        final_hidden = self.act(self.fc(torch.cat(
            (final_hidden[:, -2, :, :], final_hidden[:, -1, :, :]), dim=2)))
        final_cell = self.act(self.fc(torch.cat(
            (final_cell[:, -2, :, :], final_cell[:, -1, :, :]), dim=2)))
        # print(f'Encoder After cat:, h: {final_hidden.shape}, c: {final_cell.shape}')

        return output, final_hidden, final_cell


class grfDecoder(nn.Module):

    def __init__(self,
                 out_dim=1,  # output feature
                 hid_dim=400,
                 fc1_dim=300,
                 fc2_dim=150,
                 num_layers=3,
                 bidirectional=False,
                 dropout=0.3):
        super(grfDecoder, self).__init__()

        self.output_size = out_dim
        self.hidden_size = hid_dim
        self.fc1_size = fc1_dim
        self.fc2_size = fc2_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.output_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=self.bidirectional,
                            num_layers=self.num_layers,
                            dropout=self.dropout)
        self.fc1 = nn.Linear(self.hidden_size, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.regressor = nn.Linear(self.fc2_size, self.output_size)
        self.act = nn.Tanh()

    def forward(self, input, hidden, cell):
        '''
        INPUT
        - input: previous step output in decoder (Batch, 1)
        - hidden: previous hidden state (Layers, Batch, dec_hid_dim)
        - cell: previous cell state (Layers, Batch, dec_hid_dim)

        OUTPUT
        - output: (Batch, 1, hid_dim) -> (Batch, hid_dim) ->(Batch, fc1_dim) ->
        (Batch, fc2_dim) -> (Batch, out_dim=1)
        - hidden: (Layers, Batch, hid_dim)
        - cell: (Layers, Batch, hid_dim)
        '''
        input = input.unsqueeze(1)  # (Batch, 1, out_dim)

        # print(f'Decoder Before:, input: {input.shape}, h: {hidden.shape}, c: {cell.shape}')
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        # print(f'Decoder After:, output: {output.shape}, h: {hidden.shape}, c: {cell.shape}')
        output = self.act(self.fc1(output.squeeze(1)))  # (Batch, fc1)
        output = self.act(self.fc2(output))  # (Batch, fc2)
        output = self.regressor(output)  # (Batch, out_dim=1)
        # print(f'Decoder Final:, output: {output.shape}, h: {hidden.shape}, c: {cell.shape}')
        return output, hidden, cell


class grfSeq2Seq(nn.Module):
    # source_len=20, forecast_len=20 <- from input data
    def __init__(self,
                 device,
                 in_dim=6,
                 hid_dim=400,
                 fc1_dim=300,
                 fc2_dim=150,
                 out_dim=1,
                 num_layers=3,
                 bidirectional=True,
                 dropout=0.3):
        super(grfSeq2Seq, self).__init__()

        self.input_size = in_dim
        self.hidden_size = hid_dim
        self.fc1_size = fc1_dim
        self.fc2_size = fc2_dim
        self.output_size = out_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.device = device
        self.encoder = grfEncoder(self.input_size,
                                  self.hidden_size,
                                  self.num_layers,
                                  self.bidirectional,
                                  self.dropout)
        self.decoder = grfDecoder(self.output_size,
                                  self.hidden_size,
                                  self.fc1_size,
                                  self.fc2_size,
                                  self.num_layers,
                                  False,
                                  self.dropout)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        '''
        Forward method of LSTM Seq2Seq Module

        INPUT
        - src: source batched data (Batch, source_len, in_dim=6)
        - trg: target batched data (Batch, forecast_len, out_dim=1)

        OUTPUT
        - outputs of prediction (Batch, forecast_len, out_dim=1)
        '''
        batch_size, forecast_len, _ = trg.shape
        # Encoder (start from zero-hidden & zero-cell states)
        hidden_0 = torch.zeros(2*self.num_layers,
                               batch_size,
                               int(self.hidden_size/2.0)).to(self.device)
        cell_0 = torch.zeros(2*self.num_layers,
                             batch_size,
                             int(self.hidden_size/2.0)).to(self.device)
        output, hidden, cell = self.encoder(src, hidden_0, cell_0)

        # Decoder
        decoder_hidden = hidden
        decoder_cell = cell
        decoder_outputs = torch.zeros(forecast_len,
                                      batch_size,
                                      self.output_size).to(self.device)
        # decoder_input = trg[:, 0, :]  # (Batch, 1)
        # (Batch, 1)
        decoder_input = torch.zeros(batch_size,
                                    self.output_size).to(self.device)
        for t in range(forecast_len):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(
                decoder_input, decoder_hidden, decoder_cell)
            decoder_outputs[t] = decoder_output

            teacher_force = random.random() < teacher_forcing_ratio
            # input update (Batch, 1)
            if t < forecast_len - 1:
                decoder_input = trg[:, t+1, :] \
                    if teacher_force else decoder_output
            else:
                continue

        return decoder_outputs.permute(1, 0, 2)

    def predict(self, input, forecast_len=1):
        '''
        Predict method of LSTM Seq2Seq Module

        INPUT
        - input: (Batch, source_len, in_dim=1)
        - forecast_len: (default=1)

        OUTPUT
        - outputs of prediction (Batch, forecast_len, out_dim=1)
        '''
        batch_size, _, _ = input.shape
        # Encoder
        # Encoder (start from zero-hidden & zero-cell states)
        hidden_0 = torch.zeros(2*self.num_layers,
                               batch_size,
                               int(self.hidden_size/2.0)).to(self.device)
        cell_0 = torch.zeros(2*self.num_layers,
                             batch_size,
                             int(self.hidden_size/2.0)).to(self.device)
        output, hidden, cell = self.encoder(input, hidden_0, cell_0)

        # Decoder
        decoder_hidden = hidden
        decoder_cell = cell
        decoder_outputs = torch.zeros(forecast_len,
                                      batch_size,
                                      self.output_size).to(self.device)
        # (Batch, 1)
        decoder_input = torch.zeros(batch_size,
                                    self.output_size).to(self.device)

        for t in range(forecast_len):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(
                decoder_input, decoder_hidden, decoder_cell)
            decoder_outputs[t] = decoder_output

            # input update (Batch, 1)
            if t < forecast_len - 1:
                decoder_input = decoder_output
            else:
                continue

        return decoder_outputs.permute(1, 0, 2)


# 후에 force output에 filtering하고 GRF 추정하도록 수정하기
# GRF output이 음수가 나오지 않도록 재확인해보자
class GRF_predictor:
    def __init__(self, start_time: float,
                 report_start: float,
                 report_duration: float,
                 left_path: str,
                 right_path: str,
                 paretic_path: str,
                 nonparetic_path: str,
                 calib_model_path: str,
                 grf_model_path: str,
                 save_path: str,
                 size: str,
                 paretic_side: str,
                 body_weight: float,
                 volt_filter=int(0),
                 force_filter=int(1),
                 force_norm=int(1),
                 grf_filter=int(0)
                 ):
        '''
        Parameters
        ----------
        start_time : float
        left_path : str
            The example is
            '../../bag/log_2023-08-04-16-26-19/afo_sensor-soleSensor_left.csv'.
        right_path : str
            The example is
            '../../bag/log_2023-08-04-16-26-19/afo_sensor-soleSensor_right.csv'.
        paretic_path : str
            The example is
            '../../bag/log_2023-08-04-16-26-19/afo_detector-gait_paretic.csv'.
        nonparetic_path : str
            The example is
            '../../bag/log_2023-08-04-16-26-19/afo_detector-gait_nonparetic.csv'.
        calib_model_path : str
            The example is
            '../../model/CHAR_230815_280_LP/'.
        grf_model_path : str
            The example is
            '../../model/GRF_230815/LSTM_GRF.pt'.
        size : str
            The example is '280'
        paretic_side : str
            The example is 'L'.
        body_weight : float

        Returns
        -------
        None.

        '''
        # 17 arguments
        self.start_time = float(start_time)
        self.report_start = float(report_start)
        self.report_end = self.report_start + float(report_duration)
        self.paretic_path = paretic_path
        self.nonparetic_path = nonparetic_path
        self.left_path = left_path
        self.right_path = right_path
        self.calib_model_path = calib_model_path
        self.grf_model_path = grf_model_path
        self.paretic_side = str(paretic_side)
        self.body_weight = body_weight
        # filtering
        self.volt_filter = int(volt_filter)
        self.force_filter = int(force_filter)
        self.force_norm = int(force_norm)
        self.grf_filter = int(grf_filter)
        # force model
        # if self.calib_model_path[-1] == "/":  # LSTM
        print(self.calib_model_path)
        self.calib_model_name = "LSTM"
        self.calib_dict_name = self.calib_model_path[-19:-1]
        # else:
            # self.calib_model_name = "piecewise_linear"
            # self.json_path = self.calib_model_path
            # with open(self.json_path, 'r') as json_file:
            #     self.model_dict = json.load(json_file)
            #     json_file.close()
            # self.breakpoints = 2
            # # 35
            # # piecewise_linear_info_30N_ver2.json
            # self.calib_dict_name = self.json_path[-54:-39]
        ###########################################################
        # sample save path:
        # D://OneDrive - SNU/범부처-DESKTOP-2JL44HH/C_임상_2023/
        # 2023-08-16/report_RAW/2MWT_BARE_CueOFF_90m/
        self.save_path = save_path
        self.device = torch.device('cpu')
        self.size = str(size)
        self.sensor_num = int(6)
        self.force_input_length = int(20)
        self.grf_input_length = int(20)
        self.save_path_read()
        # Read gait detection data
        # 이 기능은 후에 class 밖으로 빼자
        # self.Gait_detection_data_read()

    def get_time_data(self):
        return np.array(self.left_time), np.array(self.right_time)

    def get_voltage_raw_data_modify(self):
        self.left_voltage = pd.DataFrame(self.left_vout)
        self.left_voltage.columns = [
            "time", "v1", "v2", "v3", "v4", "v5", "v6"
        ]
        self.right_voltage = pd.DataFrame(self.right_vout)
        self.right_voltage.columns = [
            "time", "v1", "v2", "v3", "v4", "v5", "v6"
        ]

    # def get_voltage_raw_data(self):
    #     # Read vout data
    #     self.vout_data_read()
    #     # Calib model loading
    #     self.force_model_load()
    #     # Calibration model inference
    #     self.calib_total_prediction()
    #     self.GRF_preprocessing()

    #     left_stance_time = pd.DataFrame(
    #         self.left_time.isin(self.Syncedleft_time) * 1.0
    #         )
    #     right_stance_time = pd.DataFrame(
    #         self.right_time.isin(self.Syncedright_time) * 1.0
    #         )

    #     left_stance_time.columns = ["stance time"]
    #     right_stance_time.columns = ["stance time"]

    #     left_voltage = pd.concat([self.left_data, left_stance_time], axis=1)
    #     right_voltage = pd.concat([self.right_data, right_stance_time], axis=1)

    #     return left_voltage, right_voltage

    # def get_force_raw_data(self):
    #     # Read vout data
    #     self.vout_data_read()
    #     # Calib model loading
    #     self.force_model_load()
    #     # Calibration model inference
    #     self.calib_total_prediction()
    #     self.GRF_preprocessing()

    #     left_stance_time = pd.DataFrame(
    #         self.left_time.isin(self.Syncedleft_time) * 1.0
    #         )
    #     right_stance_time = pd.DataFrame(
    #         self.right_time.isin(self.Syncedright_time) * 1.0
    #         )

    #     left_stance_time.columns = ["stance time"]
    #     right_stance_time.columns = ["stance time"]

    #     left_force = pd.concat(
    #         [self.left_time, pd.DataFrame(self.left_force), left_stance_time],
    #         axis=1
    #         )
    #     right_force = pd.concat(
    #         [self.right_time, pd.DataFrame(self.right_force), right_stance_time],
    #         axis=1
    #         )

    #     return left_force, right_force

    def get_force_raw_data_modify(self):
        left_force_data = pd.DataFrame()
        left_force_data["time"] = self.left_voltage.time
        left_force_data_add = pd.DataFrame(self.left_force)
        left_force_data = pd.concat(
            [left_force_data, left_force_data_add],
            axis=1, ignore_index=True
        )
        left_force_data.columns = [
            "time", "f1", "f2", "f3", "f4", "f5", "f6"
        ]
        right_force_data = pd.DataFrame()
        right_force_data["time"] = self.right_voltage.time
        right_force_data_add = pd.DataFrame(self.right_force)
        right_force_data = pd.concat(
            [right_force_data, right_force_data_add],
            axis=1, ignore_index=True
        )
        right_force_data.columns = [
            "time", "f1", "f2", "f3", "f4", "f5", "f6"
        ]
        self.left_force = left_force_data
        self.right_force = right_force_data

    # # Set GRF values of swing phase as zero
    # def get_GRF_raw_data(self):
    #     self.GRF_initialization()

    #     left_GRF_stance = pd.concat(
    #         [self.Syncedleft_time, pd.DataFrame(self.left_grf)], axis=1
    #         )
    #     right_GRF_stance = pd.concat(
    #         [self.Syncedright_time, pd.DataFrame(self.right_grf)], axis=1
    #         )

    #     left_GRF = self.left_time.to_frame().merge(
    #         left_GRF_stance, how='left', on="time"
    #         )
    #     left_GRF = left_GRF.fillna(0)
    #     right_GRF = self.right_time.to_frame().merge(
    #         right_GRF_stance, how='left', on="time"
    #         )
    #     right_GRF = right_GRF.fillna(0)

    #     return left_GRF, right_GRF

    def get_grf_raw_data_modify(self):
        left_grf_data = pd.DataFrame()
        left_grf_data["time"] = self.left_voltage.time
        left_grf_data["grf"] = self.left_grf

        right_grf_data = pd.DataFrame()
        right_grf_data["time"] = self.right_voltage.time
        right_grf_data["grf"] = self.right_grf

        self.left_grf = left_grf_data
        self.right_grf = right_grf_data

    def save_path_read(self):
        # Save path reading
        path_list, name_list = folder_path_name(
            self.save_path, option="start", char="RAW", T_F=1
        )
        if len(path_list) != 6:
            self.grf_initialization()
        else:
            self.saved_data_load(path_list, name_list)

    def saved_data_load(self, path_list, name_list):
        for num, name in enumerate(name_list):
            data_type = name[4:7]  # voltage, force, grf
            data_direction = name[-7:-4]
            if data_direction == 'EFT':  # left
                if data_type == 'VOL':
                    self.left_voltage = pd.read_csv(path_list[num], header=0)
                if data_type == 'FOR':
                    self.left_force = pd.read_csv(path_list[num], header=0)
                if data_type == 'GRF':
                    self.left_grf = pd.read_csv(path_list[num], header=0)
            else:
                if data_type == 'VOL':
                    self.right_voltage = pd.read_csv(path_list[num], header=0)
                if data_type == 'FOR':
                    self.right_force = pd.read_csv(path_list[num], header=0)
                if data_type == 'GRF':
                    self.right_grf = pd.read_csv(path_list[num], header=0)

    def grf_initialization(self):
        # Data process
        self.vout_data_read()
        if self.calib_model_name == "LSTM":
            self.force_model_load()
        self.calib_total_prediction()
        self.force_filtering()
        self.grf_model_load()
        self.grf_total_prediction()

        # Data load for saving
        self.get_voltage_raw_data_modify()
        self.get_force_raw_data_modify()
        self.get_grf_raw_data_modify()

        # Save dataframe
        self.left_voltage.to_csv(
            self.save_path + 'RAW_VOLTAGE_LEFT.csv',
            sep=",", header=True, index=False
        )
        self.right_voltage.to_csv(
            self.save_path + 'RAW_VOLTAGE_RIGHT.csv',
            sep=",", header=True, index=False
        )
        # self.left_force.to_csv(
        #     self.save_path + 'RAW_FORCE_%s_LEFT.csv' %
        #     self.calib_model_path[-19:-1],  # CHAR_230815_280_LP
        #     sep=",", header=True, index=False
        # )
        # self.right_force.to_csv(
        #     self.save_path + 'RAW_FORCE_%s_RIGHT.csv' %
        #     self.calib_model_path[-19:-1],  # CHAR_230815_280_LP
        #     sep=",", header=True, index=False
        # )
        self.left_force.to_csv(
            self.save_path + 'RAW_FORCE_%s_LEFT.csv' %
            self.calib_dict_name,  # CHAR_230815_280
            sep=",", header=True, index=False
        )
        self.right_force.to_csv(
            self.save_path + 'RAW_FORCE_%s_RIGHT.csv' %
            self.calib_dict_name,  # CHAR_230815_280
            sep=",", header=True, index=False
        )
        # GRF_241001/BiLSTM_GRF.pt
        self.left_grf.to_csv(
            self.save_path + 'RAW_GRF_%s_LEFT.csv' %
            (self.grf_model_path[-20:-14] + "_" +  # 241001
             self.grf_model_path[-13:-7]),  # BiLSTM
            sep=",", header=True, index=False
        )
        self.right_grf.to_csv(
            self.save_path + 'RAW_GRF_%s_RIGHT.csv' %
            (self.grf_model_path[-20:-14] + "_" +  # 241001
             self.grf_model_path[-13:-7]),  # BiLSTM
            sep=",", header=True, index=False
        )
        print("COMPLETE SAVING!")

    def vout_data_read(self):
        print("vout data read method!")
        left_data = pd.read_csv(self.left_path, header=0)
        right_data = pd.read_csv(self.right_path, header=0)
        # Column selection - time, vout only
        left_column_list = ["time",
                            "v1_L", "v2_L", "v3_L", "v4_L", "v5_L", "v6_L"]
        right_column_list = ["time",
                             "v1_R", "v2_R", "v3_R", "v4_R", "v5_R", "v6_R"]
        column_num_list = [0]
        column_num_list.extend(list(np.arange(3, 3+self.sensor_num)))
        left_data = left_data.iloc[:, column_num_list]
        right_data = right_data.iloc[:, column_num_list]
        left_data.columns = left_column_list
        right_data.columns = right_column_list
        self.left_data_columns = left_column_list
        self.right_data_columns = right_column_list
        # Time initialization with respect to start time
        left_data = left_data.astype(float)
        right_data = right_data.astype(float)
        left_data.time -= self.start_time
        right_data.time -= self.start_time
        # Data trimming with respect to time info
        left_data = left_data[
            (left_data.time >= self.report_start) &
            (left_data.time <= self.report_end)]
        right_data = right_data[
            (right_data.time >= self.report_start) &
            (right_data.time <= self.report_end)]
        # Vout data update
        self.left_data = left_data
        self.left_time = left_data.time
        self.right_data = right_data
        self.right_time = right_data.time
        self.left_vout = left_data
        self.right_vout = right_data

    def force_model_load(self):
        self.left_model = np.array([])
        self.right_model = np.array([])
        for num in np.arange(self.sensor_num):
            # LSTM model only
            leftmodel = forceSeq2Seq(
                device=self.device,
                in_dim=1,  # input size
                hid_dim=128,  # hidden size
                out_dim=1,  # output size
                num_layers=3,
                bidirectional=False,
                dropout=0.3)
            rightmodel = forceSeq2Seq(
                device=self.device,
                in_dim=1,  # input size
                hid_dim=128,  # hidden size
                out_dim=1,  # output size
                num_layers=3,
                bidirectional=False,
                dropout=0.3)
            leftmodel.load_state_dict(torch.load(
                self.calib_model_path + self.size + "Left_" +
                str(num + 1) + ".pt",
                map_location=self.device))
            rightmodel.load_state_dict(torch.load(
                self.calib_model_path + self.size + "Right_" +
                str(num + 1) + ".pt",
                map_location=self.device))

            self.left_model = np.append(self.left_model, leftmodel)
            self.right_model = np.append(self.right_model, rightmodel)

    def force_transform_LSTM(self, num, sensor_dir="L"):
        if sensor_dir == "L":
            # left_data
            if len(self.left_vout) < self.force_input_length:
                extra_row = np.repeat(
                    [self.left_vout[0]],
                    repeats=self.force_input_length - len(self.left_vout),
                    axis=0)
                input_data = np.concatenate(
                    (extra_row, self.left_vout), axis=0)
                x = input_data[
                    (len(input_data) - self.force_input_length):
                        len(input_data), (num - 1)]
            else:
                input_data = self.left_vout
                x = input_data[
                    (len(self.left_vout) - self.force_input_length):
                        len(self.left_vout), (num - 1)]
        else:
            # right_data
            if len(self.right_vout) < self.force_input_length:
                extra_row = np.repeat(
                    [self.right_vout[0]],
                    repeats=self.force_input_length - len(self.right_vout),
                    axis=0)
                input_data = np.concatenate(
                    (extra_row, self.right_vout), axis=0)
                x = input_data[(len(input_data) - self.force_input_length):
                               len(input_data), (num - 1)]
            else:
                input_data = self.right_vout
                x = input_data[
                    (len(self.right_vout) - self.force_input_length):
                        len(self.right_vout), (num - 1)]
        x = torch.from_numpy(x)
        x = x.unsqueeze(1)
        x = torch.stack([x], dim=0).float()

        return x

    def force_transform_piecewise_linear(self, num, sensor_dir="L"):
        if sensor_dir == "L":
            # left_data
            input_data = self.left_vout
        else:
            input_data = self.right_vout
        x = input_data[len(input_data) - 1, (num - 1)]

        return x

    def piecewise_linear(self, idx, input):
        # idx: 1~6
        y = 0.0
        y += self.model_dict[self.calib_dict_name][self.calib_dir]["constant"][str(idx)]
        y += self.model_dict[self.calib_dict_name][self.calib_dir]["alpha"][str(idx)][0] * input
        for b_int in np.arange(1, self.breakpoints + 1):
            y += self.model_dict[self.calib_dict_name][self.calib_dir]["alpha"][str(idx)][b_int] *\
                np.maximum(input -
                           self.model_dict[self.calib_dict_name][self.calib_dir]["breakpoint"][str(idx)][b_int-1], 0.0)
        if y < 0:
            y = 0.0

        return y

    def calib_one_prediction(self, idx, sensor_dir="L"):
        if self.calib_model_name == "LSTM":
            # LSTM model
            num_ind = 4
        else:  # piecewise linear
            num_ind = 1
        if sensor_dir == "L":
            # left_data
            # left_data indexing
            if idx <= len(self.left_data)-2:
                self.left_vout = self.left_data[0:idx+1][:]
                self.left_vout = self.left_vout.iloc[:, 1:]
                self.left_vout = np.array(self.left_vout)
            else:
                self.left_vout = self.left_data[:][1:]
                self.left_vout = np.array(self.left_vout)
            # left_data prediction
            if self.calib_model_name == "LSTM":
                # LSTM model
                _, left_name_list = folder_path_name(
                    self.calib_model_path, "include", "Left")
                left_name_list = [
                    name for name in left_name_list
                    if int(name[-num_ind]) <= self.sensor_num
                ]
                left_name_list = sorted(left_name_list,
                                        key=lambda x: int(x[-num_ind]),
                                        reverse=False)
            else:  # piecewise linear
                left_name_list = [
                    self.calib_dict_name[-3:] + "Left_1",
                    self.calib_dict_name[-3:] + "Left_2",
                    self.calib_dict_name[-3:] + "Left_3",
                    self.calib_dict_name[-3:] + "Left_4",
                    self.calib_dict_name[-3:] + "Left_5",
                    self.calib_dict_name[-3:] + "Left_6"
                    ]
                self.calib_dir = "Left"
            left_output = np.array([])

            for name in left_name_list:
                if self.calib_model_name == "LSTM":
                    # LSTM model
                    model = self.left_model[int(name[-num_ind]) - 1]
                    model.eval()
                    with torch.no_grad():
                        x = self.force_transform_LSTM(int(name[-num_ind]),
                                                      str(sensor_dir))
                        left_output = np.append(
                            left_output,
                            model.predict(input=x, forecast_len=1))
                else:  # piecewise linear
                    x = self.force_transform_piecewise_linear(
                        int(name[-num_ind]), str(sensor_dir))
                    left_output = np.append(
                        left_output,
                        self.piecewise_linear(int(name[-num_ind]), x)
                        )

            return np.expand_dims(left_output, axis=0)
        else:
            # right_data
            # right_data indexing
            if idx <= len(self.right_data)-2:
                self.right_vout = self.right_data[0:idx+1][:]
                self.right_vout = self.right_vout.iloc[:, 1:]
                self.right_vout = np.array(self.right_vout)
            else:
                self.right_vout = self.right_data[:][1:]
                self.right_vout = np.array(self.right_vout)
            # right_data prediction
            if self.calib_model_name == "LSTM":
                # LSTM model
                _, right_name_list = folder_path_name(
                    self.calib_model_path, "include", "Right")
                right_name_list = [
                    name for name in right_name_list
                    if int(name[-num_ind]) <= self.sensor_num
                ]
                right_name_list = sorted(right_name_list,
                                         key=lambda x: int(x[-num_ind]),
                                         reverse=False)
            else:  # piecewise linear
                right_name_list = [
                    self.calib_dict_name[-3:] + "right_1",
                    self.calib_dict_name[-3:] + "right_2",
                    self.calib_dict_name[-3:] + "right_3",
                    self.calib_dict_name[-3:] + "right_4",
                    self.calib_dict_name[-3:] + "right_5",
                    self.calib_dict_name[-3:] + "right_6"
                    ]
                self.calib_dir = "ight"
            right_output = np.array([])

            for name in right_name_list:
                if self.calib_model_name == "LSTM":
                    # LSTM model
                    model = self.right_model[int(name[-num_ind]) - 1]
                    model.eval()
                    with torch.no_grad():
                        x = self.force_transform_LSTM(int(name[-num_ind]),
                                                      str(sensor_dir))
                        right_output = np.append(
                            right_output,
                            model.predict(input=x, forecast_len=1))
                else:  # piecewise linear
                    x = self.force_transform_piecewise_linear(
                        int(name[-num_ind]), str(sensor_dir))
                    right_output = np.append(
                        right_output,
                        self.piecewise_linear(int(name[-num_ind]), x)
                        )

            return np.expand_dims(right_output, axis=0)

    def calib_total_prediction(self):
        print("calib total prediction method!")
        # left_data
        self.left_force = self.calib_one_prediction(idx=0, sensor_dir="L")
        for left_idx in np.arange(1, len(self.left_data)):
            self.left_force = np.append(
                self.left_force,
                self.calib_one_prediction(idx=left_idx, sensor_dir="L"),
                axis=0)
        # right_data
        self.right_force = self.calib_one_prediction(idx=0, sensor_dir="R")
        for right_idx in np.arange(1, len(self.right_data)):
            self.right_force = np.append(
                self.right_force,
                self.calib_one_prediction(
                    idx=right_idx, sensor_dir="R"), axis=0)

    def force_filtering(self):
        if self.force_filter == int(1):
            # filtering
            # (1) low-pass Butterworth filter
            sensor_freq = 100
            cutoff_freq = 10
            sensor_lp_order = 2
            for i in np.arange(self.sensor_num):
                self.left_force[:, i] = butter_lowpass_filter(
                    self.left_force[:, i],
                    cutoff_freq,
                    sensor_freq/2,
                    sensor_lp_order)
                self.right_force[:, i] = butter_lowpass_filter(
                    self.right_force[:, i],
                    cutoff_freq,
                    sensor_freq/2,
                    sensor_lp_order)

            # (2) 1D median filter
            sensor_kernel_size = int(1000/sensor_freq) + 1
            for i in np.arange(self.sensor_num):
                self.left_force[:, i] = signal.medfilt(
                    self.left_force[:, i],
                    kernel_size=sensor_kernel_size)
                self.right_force[:, i] = signal.medfilt(
                    self.right_force[:, i],
                    kernel_size=sensor_kernel_size)

            # Normalization by z-score normalization
            if self.force_norm == int(1):
                for i in np.arange(self.sensor_num):
                    self.left_force[:, i] = z_score_normalization_previous(
                        self.left_force[:, i])
                    self.right_force[:, i] = z_score_normalization_previous(
                        self.right_force[:, i])
            # Normalization by min-max normalization
            # if self.force_norm == int(1):
            #     for i in np.arange(self.sensor_num):
            #         self.left_time, self.left_vout, self.left_force =\
            #             z_score_normalization(
            #                 self.left_time, self.left_vout, self.left_force, i
            #                 )
            #         self.right_time, self.right_vout, self.right_force =\
            #             z_score_normalization(
            #                 self.right_time, self.right_vout,
            #                 self.right_force, i
            #                 )

            if len(self.left_vout) != len(self.left_force):
                self.left_time = self.left_time[:len(self.left_vout)]
                self.left_vout = self.left_vout[:len(self.left_vout), :]
                self.left_force = self.left_force[:len(self.left_vout), :]
            if len(self.right_vout) != len(self.right_force):
                self.right_time = self.right_time[:len(self.right_vout)]
                self.right_vout = self.right_vout[:len(self.right_vout), :]
                self.right_force = self.right_force[:len(self.right_vout), :]

        if self.calib_model_name == "LSTM":
            self.left_time = self.left_time[:-self.force_input_length]
            self.left_vout = self.left_vout[:-self.force_input_length, :]
            self.left_force = self.left_force[self.force_input_length:, :]

            self.right_time = self.right_time[:-self.force_input_length]
            self.right_vout = self.right_vout[:-self.force_input_length, :]
            self.right_force = self.right_force[self.force_input_length:, :]

    # def Gait_detection_data_read(self):
    #     # stance = 1, swing = 2
    #     # stance time: 1 -> 2
    #     pareticData = pd.read_csv(self.paretic_path, header=0)
    #     nonpareticData = pd.read_csv(self.nonparetic_path, header=0)
    #     # Time initialization with respect to start time
    #     pareticData = pareticData.astype(float)
    #     nonpareticData = nonpareticData.astype(float)
    #     pareticData.Time -= self.start_time
    #     nonpareticData.Time -= self.start_time
    #     # stance start timing, end timing, duration
    #     # paretic side
    #     pareticEvent_df = pd.DataFrame(
    #         columns=["start time", "end time", "duration"])
    #     par_flag = 0
    #     for par_idx in np.arange(len(pareticData)):
    #         # start time
    #         if pareticData.iloc[par_idx, 1] == 1:
    #             paretic_start = pareticData.iloc[par_idx, 0]
    #             par_flag = 1
    #         elif (pareticData.iloc[par_idx, 1] == 2) & (par_flag == 1):
    #             paretic_end = pareticData.iloc[par_idx, 0]
    #             par_flag = 2
    #         else:
    #             pass
    #         if par_flag == 2:
    #             paretic_duration = paretic_end - paretic_start
    #             paretic_add = pd.DataFrame({'start time': [paretic_start],
    #                                         'end time': [paretic_end],
    #                                         'duration': [paretic_duration]})
    #             pareticEvent_df = pd.concat([pareticEvent_df, paretic_add],
    #                                         axis=0, ignore_index=True)
    #     # nonparetic side
    #     nonpareticEvent_df = pd.DataFrame(
    #         columns=["start time", "end time", "duration"])
    #     nonpar_flag = 0
    #     for nonpar_idx in np.arange(len(nonpareticData)):
    #         # start time
    #         if nonpareticData.iloc[nonpar_idx, 1] == 1:
    #             nonparetic_start = nonpareticData.iloc[nonpar_idx, 0]
    #             nonpar_flag = 1
    #         elif (nonpareticData.iloc[nonpar_idx, 1] == 2) &\
    #              (nonpar_flag == 1):
    #             nonparetic_end = nonpareticData.iloc[nonpar_idx, 0]
    #             nonpar_flag = 2
    #         else:
    #             pass
    #         if nonpar_flag == 2:
    #             nonparetic_duration = nonparetic_end - nonparetic_start
    #             nonparetic_add = pd.DataFrame(
    #                 {'start time': [nonparetic_start],
    #                  'end time': [nonparetic_end],
    #                  'duration': [nonparetic_duration]}
    #                 )
    #             nonpareticEvent_df = pd.concat(
    #                 [nonpareticEvent_df, nonparetic_add],
    #                 axis=0,
    #                 ignore_index=True
    #                 )
    #     # df update - L, R designation
    #     if self.paretic_side == 'L':
    #         self.leftTiming_df = pareticEvent_df
    #         self.rightTiming_df = nonpareticEvent_df
    #     else:
    #         self.leftTiming_df = nonpareticEvent_df
    #         self.rightTiming_df = pareticEvent_df

    # def GRF_preprocessing(self):
    #     # stance phase만 골라서 GRF inference에 활용
    #     left_force_df = pd.DataFrame()
    #     left_force_df["time"] = self.left_time
    #     left_df_add = pd.DataFrame(self.left_force)
    #     left_force_df = pd.concat([left_force_df, left_df_add], axis=1,
    #                              ignore_index=True)
    #     left_force_df.columns = ["time",
    #                             "f1", "f2", "f3",
    #                             "f4", "f5", "f6"]
    #     right_force_df = pd.DataFrame()
    #     right_force_df["time"] = self.right_time
    #     right_df_add = pd.DataFrame(self.right_force)
    #     right_force_df = pd.concat(
    #         [right_force_df, right_df_add],
    #         axis=1,
    #         ignore_index=True
    #         )
    #     right_force_df.columns = ["time",
    #                              "f1", "f2", "f3",
    #                              "f4", "f5", "f6"]
    #     # Synced df with respect to stance timing
    #     # left
    #     for left_idx in np.arange(len(self.leftTiming_df)):
    #         if left_idx == 0:
    #             Syncedleft_force = left_force_df[
    #                 (
    #                     left_force_df.time
    #                     >= self.leftTiming_df.loc[left_idx, "start time"]
    #                 ) &
    #                 (
    #                     left_force_df.time
    #                     <= self.leftTiming_df.loc[left_idx, "end time"]
    #                 )
    #                 ]
    #         else:
    #             Syncedleft_force = pd.concat([Syncedleft_force, left_force_df[
    #                 (
    #                     left_force_df.time
    #                     >= self.leftTiming_df.loc[left_idx, "start time"]
    #                 ) &
    #                 (
    #                     left_force_df.time
    #                     <= self.leftTiming_df.loc[left_idx, "end time"]
    #                 )
    #                 ]], axis=0, ignore_index=True)
    #     self.Syncedleft_time = Syncedleft_force.iloc[:, 0]
    #     self.Syncedleft_time.reset_index(drop=True, inplace=True)
    #     self.Syncedleft_force = np.array(Syncedleft_force.iloc[:, 1:])
    #     # right
    #     for right_idx in np.arange(len(self.rightTiming_df)):
    #         if right_idx == 0:
    #             Syncedright_force = right_force_df[
    #                 (
    #                     right_force_df.time
    #                     >= self.rightTiming_df.loc[right_idx, "start time"]
    #                 ) &
    #                 (
    #                     right_force_df.time
    #                     <= self.rightTiming_df.loc[right_idx, "end time"]
    #                 )
    #                 ]
    #         else:
    #             Syncedright_force = pd.concat([Syncedright_force, right_force_df[
    #                 (
    #                     right_force_df.time
    #                     >= self.rightTiming_df.loc[right_idx, "start time"]
    #                 ) &
    #                 (
    #                     right_force_df.time
    #                     <= self.rightTiming_df.loc[right_idx, "end time"]
    #                 )
    #                 ]], axis=0, ignore_index=True)
    #     self.Syncedright_time = Syncedright_force.iloc[:, 0]
    #     self.Syncedright_time.reset_index(drop=True, inplace=True)
    #     self.Syncedright_force = np.array(Syncedright_force.iloc[:, 1:])

    def grf_model_load(self):
        if self.grf_model_path[-2:] == 'pt':
            grf_model = grfSeq2Seq(device=self.device,
                                   in_dim=self.sensor_num,  # input size
                                   hid_dim=400,  # hidden size
                                   fc1_dim=300,
                                   fc2_dim=150,
                                   out_dim=1,  # output size
                                   num_layers=3,
                                   bidirectional=True,
                                   dropout=0.3)
            grf_model.load_state_dict(
                torch.load(self.grf_model_path,
                           map_location=self.device))
            self.grf_model = grf_model
        elif self.grf_model_path[-11:-8] == 'SVR':  # SVR_
            self.grf_model = joblib.load(self.grf_model_path)
        else:  # RFR
            self.grf_model = joblib.load(self.grf_model_path)

    def SVR_GRF_transform(self):
        if self.size == '280':
            xCOP_list = np.expand_dims(
                np.array([-0.15, 0.35, 0.15, 0.55, 0.45, 0.0]), axis=1)
            yCOP_list = np.expand_dims(
                np.array([1.43, 2.2, 1.54, 2.09, 1.54, 0.0]), axis=1)
        else:  # 260
            xCOP_list = np.expand_dims(
                np.array([-0.15, 0.25, 0.15, 0.45, 0.45, 0.0]), axis=1)
            yCOP_list = np.expand_dims(
                np.array([1.215, 1.985, 1.435, 1.875, 1.435, 0.0]), axis=1)

        left_sum = np.expand_dims(
            np.sum(np.array(self.left_force), axis=1), axis=1)  # N*1
        left_xCOP = np.dot(np.array(self.left_force), xCOP_list)  # N*1
        left_yCOP = np.dot(np.array(self.left_force), yCOP_list)  # N*1
        left_xCOP /= left_sum  # N*1
        left_yCOP /= left_sum  # N*1
        left_prop = np.array(self.left_force) / left_sum  # N*6
        self.GRF_leftinput = pd.DataFrame(np.concatenate(
            (left_prop, left_sum, left_xCOP, left_yCOP), axis=1))

        right_sum = np.expand_dims(
            np.sum(np.array(self.right_force), axis=1), axis=1)  # N*1
        right_xCOP = np.dot(np.array(self.right_force), xCOP_list)  # N*1
        right_yCOP = np.dot(np.array(self.right_force), yCOP_list)  # N*1
        right_xCOP /= right_sum  # N*1
        right_yCOP /= right_sum  # N*1
        right_prop = np.array(self.right_force) / right_sum  # N*6
        self.GRF_rightinput = pd.DataFrame(np.concatenate(
            (right_prop, right_sum, right_xCOP, right_yCOP), axis=1))

    def RFR_GRF_transform(self):

        left_filtered_force = np.array(copy.deepcopy(self.left_force))
        right_filtered_force = np.array(copy.deepcopy(self.right_force))
        # filtering
        # (1) low-pass Butterworth filter
        sensor_freq = 100
        cutoff_freq = 10
        sensor_lp_order = 2
        sensor_num = 6
        for i in np.arange(sensor_num):
            left_filtered_force[:, i] = butter_lowpass_filter(
                left_filtered_force[:, i],
                cutoff_freq,
                sensor_freq/2,
                sensor_lp_order)
            right_filtered_force[:, i] = butter_lowpass_filter(
                right_filtered_force[:, i],
                cutoff_freq,
                sensor_freq/2,
                sensor_lp_order)

        # (2) 1D median filter
        sensor_kernel_size = int(1000/sensor_freq) + 1
        for i in np.arange(sensor_num):
            left_filtered_force[:, i] = signal.medfilt(
                left_filtered_force[:, i],
                kernel_size=sensor_kernel_size)
            right_filtered_force[:, i] = signal.medfilt(
                right_filtered_force[:, i],
                kernel_size=sensor_kernel_size)

        # Normalization by z-score normalization
        for i in np.arange(sensor_num):
            left_filtered_force[:, i] = z_score_normalization(
                left_filtered_force[:, i])
            right_filtered_force[:, i] = z_score_normalization(
                right_filtered_force[:, i])

        # first order, second order derivatives
        left_d_force = np.array([])
        left_dd_force = np.array([])
        right_d_force = np.array([])
        right_dd_force = np.array([])
        for i in np.arange(sensor_num):

            left_dy, left_ddy = calculate_derivative(
                np.array(self.left_time),
                left_filtered_force[:, i]
            )
            if i == 0:
                left_d_force = np.expand_dims(left_dy, axis=1)
                left_dd_force = np.expand_dims(left_ddy, axis=1)
            else:
                left_d_force = np.concatenate(
                    (left_d_force, np.expand_dims(left_dy, axis=1)), axis=1)
                left_dd_force = np.concatenate(
                    (left_dd_force, np.expand_dims(left_ddy, axis=1)), axis=1)

            right_dy, right_ddy = calculate_derivative(
                np.array(self.right_time),
                right_filtered_force[:, i]
            )
            if i == 0:
                right_d_force = np.expand_dims(right_dy, axis=1)
                right_dd_force = np.expand_dims(right_ddy, axis=1)
            else:
                right_d_force = np.concatenate(
                    (right_d_force, np.expand_dims(right_dy, axis=1)), axis=1)
                right_dd_force = np.concatenate(
                    (right_dd_force, np.expand_dims(right_ddy, axis=1)), axis=1)

        self.GRF_leftinput = pd.DataFrame(np.concatenate(
            (left_filtered_force, left_d_force, left_dd_force), axis=1))
        self.GRF_rightinput = pd.DataFrame(np.concatenate(
            (right_filtered_force, right_d_force, right_dd_force), axis=1))

    def grf_transform_BiLSTM(self, idx, sensor_dir):
        if sensor_dir == 'L':
            if (int(idx) + 1) < self.grf_input_length:
                for x_num in np.arange(self.sensor_num):
                    if x_num == 0:
                        x = np.append(
                            np.array(self.left_force)[0, x_num]
                            * np.ones(
                                (1, self.grf_input_length - int(idx) - 1)),
                            np.array(self.left_force)[0:int(idx) + 1, x_num])
                        x = x.reshape(self.grf_input_length, 1)
                    else:
                        x_temp = np.append(
                            np.array(self.left_force)[0, x_num]
                            * np.ones(
                                (1, self.grf_input_length - int(idx) - 1)),
                            np.array(self.left_force)[0:int(idx) + 1, x_num])
                        x_temp = x_temp.reshape(self.grf_input_length, 1)
                        x = np.append(x, x_temp, axis=1)
            else:
                x = np.array(self.left_force)[
                    int(idx) + 1 - self.grf_input_length:int(idx) + 1, :]
        else:
            if (int(idx) + 1) < self.grf_input_length:
                for x_num in np.arange(self.sensor_num):
                    if x_num == 0:
                        x = np.append(
                            np.array(self.right_force)[0, x_num]
                            * np.ones(
                                (1, self.grf_input_length - int(idx) - 1)),
                            np.array(self.right_force)[0:int(idx) + 1, x_num])
                        x = x.reshape(self.grf_input_length, 1)
                    else:
                        x_temp = np.append(
                            np.array(self.right_force)[0, x_num]
                            * np.ones(
                                (1, self.grf_input_length - int(idx) - 1)),
                            np.array(self.right_force)[0:int(idx) + 1, x_num])
                        x_temp = x_temp.reshape(self.grf_input_length, 1)
                        x = np.append(x, x_temp, axis=1)
            else:
                x = np.array(self.right_force)[
                    int(idx) + 1 - self.grf_input_length:int(idx) + 1, :]
        x = torch.from_numpy(x)
        x = torch.stack([x], dim=0).float()

        return x

    def grf_one_prediction(self, idx, sensor_dir):
        model = self.grf_model
        model.eval()
        with torch.no_grad():
            x = self.grf_transform_BiLSTM(idx, sensor_dir)
            output = model.predict(
                input=x, forecast_len=1).detach().cpu().numpy()
            output = output.squeeze(0)[0]

        return output

    def grf_total_prediction(self):
        print("grf total prediction method!")
        if self.grf_model_path[-2:] == 'pt':
            self.left_grf = np.array([])
            self.right_grf = np.array([])
            # LEFT
            for g_ind in np.arange(len(np.array(self.left_force))):
                # prediction
                left_output = self.grf_one_prediction(
                    idx=g_ind, sensor_dir='L')
                # post-processing
                # left_output /= np.sum(np.array(self.left_force)[g_ind])
                left_output *= (self.body_weight * 9.807)
                self.left_grf = np.append(
                    self.left_grf, left_output, axis=0)
            # RIGHT
            for g_ind in np.arange(len(np.array(self.right_force))):
                # prediction
                right_output = self.grf_one_prediction(
                    idx=g_ind, sensor_dir='R')
                # post-processing
                # right_output /= np.sum(np.array(self.right_force)[g_ind])
                right_output *= (self.body_weight * 9.807)
                self.right_grf = np.append(
                    self.right_grf, right_output, axis=0)

            # post processing
            self.left_time = self.left_time[:len(self.left_vout)]
            self.left_force = self.left_force[:len(self.left_vout), :]
            self.left_grf = self.left_grf[:len(self.left_vout)]

            self.right_time = self.right_time[:len(self.right_vout)]
            self.right_force = self.right_force[:len(self.right_vout), :]
            self.right_grf = self.right_grf[:len(self.right_vout)]

            # 미루기
            self.left_time = self.left_time[:-self.grf_input_length]
            self.left_vout = self.left_vout[:-self.grf_input_length, :]
            self.left_force = self.left_force[:-self.grf_input_length, :]
            self.left_grf = self.left_grf[self.grf_input_length:]

            self.right_time = self.right_time[:-self.grf_input_length]
            self.right_vout = self.right_vout[:-self.grf_input_length, :]
            self.right_force = self.right_force[:-self.grf_input_length, :]
            self.right_grf = self.right_grf[self.grf_input_length:]

        elif self.grf_model_path[-11:-8] == 'SVR':  # SVR_
            # preprocessing
            self.SVR_GRF_transform()

            self.left_grf = np.array(
                self.grf_model.predict(self.GRF_leftinput)) *\
                (self.body_weight * 9.807)
            self.right_grf = np.array(
                self.grf_model.predict(self.GRF_rightinput)) *\
                (self.body_weight * 9.807)
        else:
            # preprocessing
            self.RFR_GRF_transform()

            self.left_grf = np.array(
                self.grf_model.predict(self.GRF_leftinput)) *\
                (self.body_weight * 9.807)
            self.right_grf = np.array(
                self.grf_model.predict(self.GRF_rightinput)) *\
                (self.body_weight * 9.807)

    # def GRF_trimmed_by_gait(self):
    #     # trimmed with respect to stance timing
    #     self.leftGRFtrimmed = pd.DataFrame(columns=['impulse', 'maximum'])
    #     self.rightGRFtrimmed = pd.DataFrame(columns=['impulse', 'maximum'])
    #     # left
    #     for left_idx in np.arange(len(self.leftTiming_df)):
    #         # index determination
    #         index_list = list(
    #             self.Syncedleft_time[
    #                 (
    #                     self.Syncedleft_time
    #                     >= self.leftTiming_df.loc[left_idx, "start time"]
    #                 ) &
    #                 (
    #                     self.Syncedleft_time
    #                     <= self.leftTiming_df.loc[left_idx, "end time"]
    #                 )
    #                 ].index)
    #         # impulse calculation
    #         GRF_leftImpulse = 0
    #         for L_idx in index_list[:-1]:
    #             delta_time = self.Syncedleft_time.loc[L_idx+1] -\
    #                 self.Syncedleft_time.loc[L_idx]
    #             GRF_value = float(self.left_grf[L_idx])
    #             GRF_leftImpulse += delta_time * GRF_value
    #         # maximum calculation
    #         GRF_leftMaximum = max(self.left_grf[index_list])
    #         # df update
    #         left_update_df = pd.DataFrame({'impulse': [GRF_leftImpulse],
    #                                        'maximum': [GRF_leftMaximum]})
    #         self.leftGRFtrimmed = pd.concat([self.leftGRFtrimmed,
    #                                          left_update_df], axis=0,
    #                                         ignore_index=True)
    #     # right
    #     for right_idx in np.arange(len(self.rightTiming_df)):
    #         # index determination
    #         index_list = list(
    #             self.Syncedright_time[
    #                 (
    #                     self.Syncedright_time
    #                     >= self.rightTiming_df.loc[right_idx, "start time"]
    #                 ) &
    #                 (
    #                     self.Syncedright_time
    #                     <= self.rightTiming_df.loc[right_idx, "end time"]
    #                 )
    #                 ].index)
    #         # impulse calculation
    #         GRF_rightImpulse = 0
    #         for R_idx in index_list[:-1]:
    #             delta_time = self.Syncedright_time.loc[R_idx+1] -\
    #                 self.Syncedright_time.loc[R_idx]
    #             GRF_value = float(self.right_grf[R_idx])
    #             GRF_rightImpulse += delta_time * GRF_value
    #         # maximum calculation
    #         GRF_rightMaximum = max(self.right_grf[index_list])
    #         # df update
    #         right_update_df = pd.DataFrame({'impulse': [GRF_rightImpulse],
    #                                        'maximum': [GRF_rightMaximum]})
    #         self.rightGRFtrimmed = pd.concat([self.rightGRFtrimmed,
    #                                          right_update_df], axis=0,
    #                                          ignore_index=True)


##
# GRF_predictor class에 save_path 추가해서 쭉 돌려놓기!
class ClinicalIndexMH:
    @staticmethod
    def get_symmetry_index_grf(
            start_time, left_path, right_path,
            paretic_path, non_paretic_path,
            model_path_calib, model_path_grf,
            raw_data_save_path, cycle_timeseries_data_save_path,
            size='280', paretic_side='L',
            body_weight=float(100), ignore_cycle=(None, None)
    ):
        grf_class = GRF_predictor(
            start_time=float(start_time),
            left_path=left_path,
            right_path=right_path,
            paretic_path=paretic_path,
            nonparetic_path=non_paretic_path,
            calib_model_path=model_path_calib,
            grf_model_path=model_path_grf,
            save_path=raw_data_save_path,
            size=size,
            paretic_side=paretic_side,
            body_weight=body_weight
        )
        if paretic_side == 'L':
            paretic_data = grf_class.left_grf
            non_paretic_data = grf_class.right_grf
        else:
            paretic_data = grf_class.right_grf
            non_paretic_data = grf_class.left_grf

        max_array, impulse_array = DataProcess.data_process(
            paretic_data,
            non_paretic_data,
            paretic_path,
            non_paretic_path,
            save_path=cycle_timeseries_data_save_path,
            data_label="grf [N]",
            title_label="GRF",
            ignore_cycle=ignore_cycle,
            start_time=start_time,
            max_flag=True,
            impulse_flag=True
        )

        return max_array, impulse_array

    @staticmethod
    def get_symmetry_index_stanceTime(start_time,
                                      paretic_path, non_paretic_path,
                                      ignore_cycle=(None, None)):

        df_paretic_gait = DataProcess.read_data_file_by_path(
            paretic_path)
        df_non_paretic_gait = DataProcess.read_data_file_by_path(
            non_paretic_path)

        df_paretic_gait.iloc[:, 0] -= start_time
        df_non_paretic_gait.iloc[:, 0] -= start_time

        df_paretic_gait = \
            get_ignored_cycle(df_paretic_gait, ignore_cycle)
        df_non_paretic_gait = \
            get_ignored_cycle(df_non_paretic_gait, ignore_cycle)

        [_, _, _, _, _, _,
         paretic_mean, paretic_stdev,
         non_paretic_mean, non_paretic_stdev] =\
            DataProcess.gait_phase_pre_processing(
                df_paretic_gait, df_non_paretic_gait)

        symmetry = paretic_mean / (paretic_mean + non_paretic_mean) * 100

        return [paretic_mean, paretic_stdev,
                non_paretic_mean, non_paretic_stdev, symmetry]


if __name__ == "__main__":
    start_run_time = time.time()
    bag_list = [
        "log_2023-08-16-15-40-08",
        "log_2023-08-16-15-41-19",
        "log_2023-08-16-15-43-10",
        "log_2023-08-16-15-44-06",
        "log_2023-08-16-15-46-14",
        "log_2023-08-16-15-51-39",
        "log_2023-08-16-15-56-59",
        "log_2023-08-16-16-02-17",
        "log_2023-08-16-16-23-57",
        "log_2023-08-16-16-27-28",
        "log_2023-08-16-16-29-04"
    ]
    session_list = [
        "10MWT_OFF_CueOFF",
        "10MWT_ON_CueOFF",
        "10MWT_ON_CueON_1",
        "10MWT_ON_CueON_2",
        "10MWT_ON_CueON_3",
        "2MWT_OFF_CueOFF_89.4m",
        "2MWT_ON_CueOFF_88.2m",
        "2MWT_ON_CueON_64.2m",
        "2MWT_BARE_CueOFF_90m",
        "10MWT_BARE_CueOFF_1",
        "10MWT_BARE_CueOFF_2"
    ]

    base_path =\
        'D:/OneDrive - SNU/범부처-DESKTOP-2JL44HH/C_임상_2023/2023-08-16/'
    calib_folder_name = 'CHAR_230815_280_LP'
    GRF_model_name = 'GRF_230815/LSTM_GRF.pt'
    # 장비 무게 포함
    body_weight = 85.1
    paretic_side = 'L'
    size = '280'

    for bag_ind in np.arange(len(bag_list)):

        bag_name = bag_list[bag_ind]
        session_name = session_list[bag_ind]

        # Read bag file
        path = base_path + 'bag/' + bag_name + '.bag'
        # bag_name[4:-9] = 2023-08-16
        save_folder = base_path + 'report/' + session_name + '/'
        create_folder(save_folder)

        bag = bagreader(path)
        start_time = bag.start_time

        # To filter specific topics with interests
        TOPIC_MH = (
            "/afo_sensor/soleSensor_left",
            "/afo_sensor/soleSensor_right",
            "/afo_detector/gait_paretic",
            "/afo_detector/gait_nonparetic"
        )

        # Definition of Clinical Indices
        left_sole_path = ""
        right_sole_path = ""
        paretic_gait_path = ""
        non_paretic_gait_path = ""

        # calib_model_path example = ../../model/CHAR_230815_280_LP/
        # GRF_model_path example = ../../model/GRF_230815/LSTM_GRF.pt
        calib_model_path = '../../model/' + calib_folder_name + '/'
        GRF_model_path = '../../model/' + GRF_model_name

        # Read Topics and calculate clinical index
        for topic in bag.topics:
            msg_topic = bag.message_by_topic(topic)

            # Use own module and methods
            if topic == TOPIC_MH[0]:
                left_sole_path = msg_topic
            elif topic == TOPIC_MH[1]:
                right_sole_path = msg_topic
            elif topic == TOPIC_MH[2]:
                paretic_gait_path = msg_topic
            elif topic == TOPIC_MH[3]:
                non_paretic_gait_path = msg_topic

        raw_data_path = '../../data/report/2023-08-16/' + session_name + '/'
        cycle_timeseries_data_save_path =\
            '../../graph/2023-08-16/' + session_name + '/'
        # create directory
        create_folder(raw_data_path)
        create_folder(cycle_timeseries_data_save_path)

        start_inference_time = time.time()

        ClinicalIndexMH.get_symmetry_index_grf(
            start_time=float(start_time),
            left_path=left_sole_path,
            right_path=right_sole_path,
            paretic_path=paretic_gait_path,
            non_paretic_path=non_paretic_gait_path,
            model_path_calib=calib_model_path,
            model_path_grf=GRF_model_path,
            raw_data_save_path=raw_data_path,
            cycle_timeseries_data_save_path=cycle_timeseries_data_save_path,
            size=size,
            paretic_side=paretic_side,
            body_weight=body_weight,
            ignore_cycle=(None, None)
        )
        # grf_class = GRF_predictor(
        #     start_time=float(start_time),
        #     left_path=left_sole_path,
        #     right_path=right_sole_path,
        #     paretic_path=paretic_gait_path,
        #     nonparetic_path=non_paretic_gait_path,
        #     calib_model_path=calib_model_path,
        #     grf_model_path=GRF_model_path,
        #     save_path=save_folder,
        #     size=size,
        #     paretic_side=paretic_side,
        #     body_weight=body_weight
        #     )
        end_inference_time = time.time()
        print("Inference time: ", end_inference_time - start_inference_time)
    end_run_time = time.time()
    print("Running time: ", end_run_time - start_run_time)

    # raw data plotting
