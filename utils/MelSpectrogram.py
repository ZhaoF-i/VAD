import numpy as np
import torchaudio
import torch

class Mel(torch.nn.Module):
    def __init__(self, window = 400, shift = 200, log = False):
        super(Mel, self).__init__()

        self.window = window
        self.shift = shift
        self.low_freq = 20
        self.high_freq = 7600
        self.log = log

    def OSD(self, mel):
        return mel

    def forward(self, input):
        Mel = torchaudio.transforms.MelSpectrogram(win_length=self.window, hop_length=self.shift,
                                                   f_min=self.low_freq, f_max=self.high_freq)(input.cpu())

        if self.log:
            Mel = torch.log(Mel)

        Mel = torch.Tensor(Mel).cuda()
        Mel = self.OSD(Mel)

        return Mel


