import numpy as np
import torchaudio
import torch

class Mel(torch.nn.Module):
    def __init__(self, n_mels = 128, window = 400, shift = 200, log = False):
        super(Mel, self).__init__()

        self.n_mels = n_mels
        self.window = window
        self.shift = shift
        self.low_freq = 20
        self.high_freq = 7600
        self.log = log

    def OSD(self, mel):
        mel_len = mel.size()[2]
        net_inp = []
        index = 0
        while True:
            if index+150 >= mel_len:
                zero = torch.zeros((1, 100, 150 - (mel_len - index)))
                one_win = mel[:,:, index: mel_len]
                one_win = torch.cat((zero, one_win), 2)
                net_inp.append(one_win.T)
                break
            one_win = mel[:,:, index: index+150]
            index += 50
            net_inp.append(one_win.T)

        net_inp = torch.cat(net_inp)
        return net_inp

    def forward(self, input):
        # Mel = torchaudio.transforms.MelSpectrogram(win_length=self.window, hop_length=self.shift,
        #                                            f_min=self.low_freq, f_max=self.high_freq)(input.cpu())
        Mel = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels , win_length=self.window, hop_length=self.shift)(input.cpu())

        if self.log:
            Mel = torch.log(Mel)

        # net_inp = self.OSD(Mel)
        Mel = torch.Tensor(Mel).cuda()

        return Mel


