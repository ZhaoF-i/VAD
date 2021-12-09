import argparse
import os
import yaml
import soundfile as sf
import logging as log

from pathlib import Path
from torch import nn
from metrics_classify import Metrics
# from metrics import Metrics
from networks.CRN import NET_Wrapper
from utils.Checkpoint import Checkpoint
from utils.progressbar import progressbar as pb
from utils.stft_istft import STFT
from utils.util import makedirs, gen_list
from torch.autograd.variable import *
import numpy as np
from utils.util import frame_level_label


class Test(object):
    def __init__(self, inpath, outpath, type='online', suffix='mix.wav'):
        self.inpath = inpath
        self.outpath = outpath
        self.type = type
        self.suffix = suffix
        self.STFT = STFT(config['WIN_LEN'], config['WIN_OFFSET']).cuda()

    def forward(self, network):
        network.eval()
        # tt_lst = gen_list(self.inpath + '/seg_wav', self.suffix)
        tt_lst = os.listdir(self.inpath + 'seg_wav')
        tt_lst.sort()
        test_label_lst = os.listdir(self.inpath + 'seg_label')
        test_label_lst.sort()
        tt_len = len(tt_lst)
        pbar = pb(0, tt_len)
        pbar.start()
        for i in range(tt_len):
            pbar.update_progress(i, 'test', '')
            mix, fs = sf.read(self.inpath + 'seg_wav/' + tt_lst[i])
            mix = mix[:,0]
            alpha_pow = 1 / (np.sqrt(np.sum(mix ** 2)) / (mix.size) + 1e-7)
            mix = mix * alpha_pow
            mixture = Variable(torch.FloatTensor(mix.astype('float32')))
            mixture = mixture.unsqueeze(0)


            """------------------------------------modify  area------------------------------------"""
            with torch.no_grad():
                est = network(mixture)
            est = est.cpu().numpy()
            """------------------------------------modify  area------------------------------------"""

            path = tt_lst[i].split('.')[0]
            np.save(self.outpath + path + '.npy', est)
            test_label = np.load(self.inpath + 'seg_label/' + test_label_lst[i])
            # test_label = frame_level_label(test_label)
            np.save(self.outpath + path + '_label.npy', test_label)


        pbar.finish()


if __name__ == '__main__':
    """
        environment part
        """
    # loading argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="trained model name, retrain if no input", default='none')
    parser.add_argument("-y", "--yaml_name", help="config file name")
    parser.add_argument("-n", "--new_test",
                        help="generate new test date[1] or only compute metrics with exist data[0]",default=0,type=int)
    args = parser.parse_args()

    # loading config
    _abspath = Path(os.path.abspath(__file__)).parent
    _project = _abspath.stem
    _yaml_path = os.path.join(_abspath, 'configs/' + args.yaml_name)
    with open(_yaml_path, 'r') as f_yaml:
        config = yaml.load(f_yaml, Loader=yaml.FullLoader)

    # if online test
    _outpath = config['OUTPUT_DIR'] + _project + config['WORKSPACE']
    # if offline test
    # _outpath = config['OFFLINE_TEST_DIR'] + _project + config['WORKSPACE']
    outpath = _outpath + '/estimations'
    makedirs([outpath])

    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_ID']
    if args.new_test:
        network = NET_Wrapper(config['WIN_LEN'], config['WIN_OFFSET'])
        network = nn.DataParallel(network)
        network.cuda()

        checkpoint = Checkpoint()
        checkpoint.load(args.model_name)
        network.load_state_dict(checkpoint.state_dict)
        log.info('#' * 14 + 'Finish Resume Model For Test' + '#' * 14)
        print(checkpoint.best_loss)

        # set type and suffix for local test dat
        inpath = config['TEST_PATH']
        test = Test(inpath=inpath, outpath=outpath, type='online', suffix='mix.wav')
        test.forward(network)

    # cal metrics
    metrics = Metrics(outpath)
    metrics.forward()
