import librosa
import torch
import torchaudio.transforms


class MFCC(torch.nn.Module):
    def __init__(self, filter_length=400, hop_length=200):
        super(MFCC, self).__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.low_freq = 20
        self.high_freq = 7600
        self.n_mels = 30
        self.kwargs = {
            'win_length': self.filter_length,
            'n_mels': self.n_mels,
            'f_min': self.low_freq,
            'f_max': self.high_freq
        }



    def forward(self, input_data):
        mfcc = torchaudio.transforms.MFCC(n_mfcc=self.n_mels, melkwargs=self.kwargs)(input_data.cpu())
        mfcc = torch.Tensor(mfcc).cuda()

        return mfcc
        # mfcc = librosa.feature.mfcc()
