import soundfile as sf
from joblib import Parallel, delayed
from pypesq import pesq  # 和matlab有0.005左右的差距  pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
from pystoi import stoi  # pip install pystoi
from pathlib import Path
from tqdm import tqdm
import numpy as np
from utils.util import frame_level_label

N_JOBS = 20
noise_type = ['babble', 'caffe']


class Metrics(object):
    def __init__(self, tt_path):
        self.tt_path = tt_path

    def getWavLst(self):
        lst = list(Path(self.tt_path).rglob('*_label.npy'))
        wavlst = [line.stem for line in lst]
        length = len(lst)
        return wavlst, length

    def forward(self):
        wavlst, length = self.getWavLst()
        wavlst.sort()

        total_error_lst = []
        print('计算误差...')
        for i in tqdm(range(length)):
            est_path = wavlst[i].split('_label')[0] + '.npy'
            est = np.load(self.tt_path + est_path)

            groud = np.load(self.tt_path + wavlst[i] + '.npy')
            groud = frame_level_label(groud, 400, 200)

            est_lst = []
            for j in range(est.shape[-1]):
                est_lst.append(est[0,:,j].argmax())

            est_lst = np.array(est_lst)

            frame_error = np.sum(est_lst!=groud)
            total_error_lst.append(frame_error)

        avg_error_num = np.array(total_error_lst).mean()
        print('平均帧错误数：' + str(avg_error_num))
        avg_error_prob = (np.array(total_error_lst) / est.shape[-1]).mean()
        print('平均帧错误率：' + str(avg_error_prob*100) + '%')

