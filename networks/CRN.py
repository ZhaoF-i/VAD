import torch.nn as nn
import torch.autograd.variable
import torchaudio


from torch.autograd.variable import *

from utils.stft_istft import STFT

from utils.MelSpectrogram import Mel


class NET_Wrapper(nn.Module):
    def __init__(self,win_len,win_offset):
        self.win_len = win_len
        self.win_offset = win_offset
        super(NET_Wrapper, self).__init__()
        self.lstm_input_size = 64 * 7
        self.lstm_layers = 2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 3), stride=(1, 2))
        self.conv1_relu = nn.ELU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 3), stride=(1, 2))
        self.conv2_relu = nn.ELU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.conv3_relu = nn.ELU()
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 3), stride=(1, 2))
        # self.conv4_relu = nn.ELU()
        # self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 3), stride=(1, 2))
        # self.conv5_relu = nn.ELU()
        # self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(2, 3), stride=(1, 2))
        # self.conv6_relu = nn.ELU()
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_input_size,
                            num_layers=self.lstm_layers,
                            batch_first=True,
                            bidirectional=True)

        self.Aver_pooling = nn.AvgPool2d((1, 7))
        self.linear_layer = nn.Sequential(nn.Linear(128, 3),
                                          nn.Dropout(0.5),
                                          nn.LeakyReLU())
        self.softmax = nn.Softmax(dim=2)


        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.pad = nn.ConstantPad2d((0, 0, 1, 0), value=0.)

        # self.STFT = STFT(self.win_len, self.win_offset).cuda()
        self.mel = torchaudio.transforms.MelSpectrogram(n_mels=64)
        # self.Mel = Mel(64 ,400, 200, False).cuda()

    def forward(self, input_data_c1):
        input=input_data_c1.unsqueeze(1)
        # input = input_data_c1.permute(0, 2, 1)
        mel_feature = self.mel(input)
        # mel_feature=mel_feature.unsqueeze(1)
        input_feature = mel_feature.permute(0, 1, 3, 2)

        e1 = self.conv1_relu(self.conv1_bn(self.conv1(self.pad(input_feature))))
        e2 = self.conv2_relu(self.conv2_bn(self.conv2(self.pad(e1))))
        e3 = self.conv3_relu(self.conv3_bn(self.conv3(self.pad(e2))))

        self.lstm.flatten_parameters()
        out_real = e3.contiguous().transpose(1, 2)
        out_real = out_real.contiguous().view(out_real.size(0), out_real.size(1), -1)
        lstm_out, _ = self.lstm(out_real)
        lstm_out_real = lstm_out.contiguous().view(lstm_out.size(0), lstm_out.size(1), 64*2, -1)
        lstm_out_real = lstm_out_real.contiguous().transpose(1, 2)  #1,64,1001,7

        out = self.Aver_pooling(lstm_out_real)    #1,64,1001,1
        out = torch.squeeze(out, 3)               #1,64,1001
        out = out.permute(0, 2, 1)                #1,1001,64

        out = self.linear_layer(out)
        out = self.softmax(out)
        out = out.permute(0, 2, 1)


        return out  # batch_size, classify, frame
