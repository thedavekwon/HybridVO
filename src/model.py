import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.nn.init import kaiming_normal_
from constants import *

# attention layer code inspired from: https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/4
class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(
            torch.Tensor(1, hidden_size), requires_grad=True
        )

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
        # apply attention layer
        weights = torch.bmm(
            inputs,
            self.att_weights.permute(1, 0)  # (1, hidden_size)  # (hidden_size, 1)
            .unsqueeze(0)  # (1, hidden_size, 1)
            .repeat(batch_size, 1, 1),  # (batch_size, hidden_size, 1)
        )
        masked = torch.softmax(F.relu(weights.squeeze()), dim=-1)
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        attentions = masked.div(_sums)
        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        return weighted, attentions


class DeepVO(nn.Module):
    def __init__(self):
        super(DeepVO, self).__init__()

        self.conv1 = conv(6, 64, 7, 2, 3, 0.2, True)
        self.conv2 = conv(64, 128, 5, 2, 2, 0.2, True)
        self.conv3 = conv(128, 256, 5, 2, 2, 0.2, True)
        self.conv3_1 = conv(256, 256, 3, 1, 1, 0.2, True)
        self.conv4 = conv(256, 512, 3, 2, 1, 0.2, True)
        self.conv4_1 = conv(512, 512, 3, 1, 1, 0.2, True)
        self.conv5 = conv(512, 512, 3, 2, 1, 0.2, True)
        self.conv5_1 = conv(512, 512, 3, 1, 1, 0.2, True)
        self.conv6 = conv(512, 1024, 3, 2, 1, 0.2, True)

        self.rnn = nn.LSTM(
            input_size=3 * 10 * 1024, hidden_size=1000, num_layers=2, batch_first=True
        )
        self.rnn_drop = nn.Dropout(0.5)
        self.linear = nn.Linear(1000, 6)

        # initalization from https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/model.py
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                kaiming_normal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                n = m.bias_hh_l0.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l0.data[start:end].fill_(1.0)

                kaiming_normal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l1.data[start:end].fill_(1.0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        x = self.flow(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.rnn(x)
        x = self.rnn_drop(x)
        x = self.linear(x)
        return x

    def flow(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.conv4(x)
        x = self.conv4_1(x)
        x = self.conv5(x)
        x = self.conv5_1(x)
        x = self.conv6(x)
        return x

    def get_loss(self, seq, pos, ang):
        pos = pos[:, 1:, :]
        ang = ang[:, 1:, :]
        y_hat = self.forward(seq)
        pos_loss = nn.functional.mse_loss(y_hat[:, :, 3:], pos)
        ang_loss = nn.functional.mse_loss(y_hat[:, :, :3], ang)
        return 100 * ang_loss + pos_loss

    def load_pretrained_flow(self, device, pretrained_path=PRETRAIN_FLOW_PATH):
        pretrained_flownet = torch.load(pretrained_path, map_location=device)
        current_state_dict = self.state_dict()
        update_state_dict = {}
        for k, v in pretrained_flownet["state_dict"].items():
            if k in current_state_dict.keys():
                update_state_dict[k] = v
        current_state_dict.update(update_state_dict)
        self.load_state_dict(current_state_dict)

    def save(self, epoch, optimizer, weight_folder=WEIGHT_FOLDER):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"{weight_folder}/{epoch}.weights",
        )


class HybridVO(nn.Module):
    def __init__(self, weight=1.0):
        super(HybridVO, self).__init__()

        self.weight = weight
        self.conv1 = conv(6, 64, 7, 2, 3, 0.2, True)
        self.conv2 = conv(64, 128, 5, 2, 2, 0.2, True)
        self.conv3 = conv(128, 256, 5, 2, 2, 0.2, True)
        self.conv3_1 = conv(256, 256, 3, 1, 1, 0.2, True)
        self.conv4 = conv(256, 512, 3, 2, 1, 0.2, True)
        self.conv4_1 = conv(512, 512, 3, 1, 1, 0.2, True)
        self.conv5 = conv(512, 512, 3, 2, 1, 0.2, True)
        self.conv5_1 = conv(512, 512, 3, 1, 1, 0.2, True)
        self.conv6 = conv(512, 1024, 3, 2, 1, 0.2, True)

        self.rnn = nn.LSTM(
            input_size=3 * 10 * 1024,
            hidden_size=500,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = Attention(1000, batch_first=True)
        self.rnn_drop = nn.Dropout(0.5)
        self.linear = nn.Linear(1000, 6)

        # initalization from https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/model.py
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                kaiming_normal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                n = m.bias_hh_l0.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l0.data[start:end].fill_(1.0)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        x = self.flow(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.rnn(x)  # [batch, seq, hidden]
        x, _ = self.attn(x)  # [batch, hidden]
        x = self.rnn_drop(x)
        x = self.linear(x)
        return x

    def flow(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.conv4(x)
        x = self.conv4_1(x)
        x = self.conv5(x)
        x = self.conv5_1(x)
        x = self.conv6(x)
        return x

    def get_loss(self, seq, pos, ang):
        ang = ang[:, 1:, :]
        y_hat = self.forward(seq)
        ang_loss = nn.functional.mse_loss(y_hat, ang)
        return ang_loss

    def load_pretrained_flow(self, device, pretrained_path=PRETRAIN_FLOW_PATH):
        pretrained_flownet = torch.load(pretrained_path, map_location=device)
        current_state_dict = self.state_dict()
        update_state_dict = {}
        for k, v in pretrained_flownet["state_dict"].items():
            if k in current_state_dict.keys():
                update_state_dict[k] = v
        current_state_dict.update(update_state_dict)
        self.load_state_dict(current_state_dict)

    def save(self, epoch, optimizer, weight_folder=WEIGHT_FOLDER):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"{weight_folder}/{epoch}.weights",
        )


def conv(in_channel, out_channel, kernel_size, stride, padding, dropout, bn):
    if bn:
        return nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
