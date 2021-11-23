from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import *
import torch
from glob import glob
from tqdm import tqdm
from random import shuffle, sample, randint, choices
from speechbrain.dataio.dataio import read_audio, write_audio
from speechbrain.lobes.augment import TimeDomainSpecAugment
from speechbrain.lobes.augment import EnvCorrupt
from functools import reduce
import torchaudio
import numpy as np


def frame_level_label(label_dict):
    frame_level_label = []
    label_dict = np.pad(label_dict, (200, 200), 'constant', constant_values=(0,0))

    counts = np.bincount(label_dict[200: 400])
    frame_level_label.append(np.argmax(counts))

    index = 160
    for i in range(1, int(label_dict.size / 160) - 2):
        counts = np.bincount(label_dict[index: index+400])
        index += 160
        frame_level_label.append(np.argmax(counts))

    counts = np.bincount(label_dict[index: index+200])
    frame_level_label.append(np.argmax(counts))

    return np.array(frame_level_label)


class Dataset(Dataset):
    def __init__(self, mode='train'):
        super().__init__()

        # 数据集音频和标签地址
        data_list = sorted(glob('/data01/spj/asr_dataset/ai_shell4_vad/TRAIN/seg_wav/*.wav'))
        lab_list = sorted(glob('/data01/spj/asr_dataset/ai_shell4_vad/TRAIN/seg_label/*.npy'))

        # 数据比例
        train_percent = 0.6
        eval_percent = 0.1
        test_percent = 0.3
        len_data = len(data_list)

        # 初始化音频、标签和字典
        self.src, self.trg = [], []

        pack = list(zip(data_list, lab_list))
        shuffle(pack)

        # 训练集or测试集
        if mode == 'validate':
            pack = pack[int(len_data * train_percent): int(len_data * (train_percent + eval_percent))]
        elif mode == 'train':
            # pack = pack[: int(len_data * train_percent)]
            pack = pack[: int(10)]
        else:
            pack = pack[int(len_data * (train_percent + eval_percent)) :]


        # 读取音频和标签
        print('Read audio and labels:')
        for p,q in tqdm(pack):
            wav_data = read_audio(p)
            # wav_data = 1 / (np.sqrt(np.sum(wav_data.numpy()) ** 2) / (wav_data.size()[0] * wav_data.size()[1]) + 1e-7)

            label_dict = np.load(q)
            label_dict = np.minimum(label_dict, 2)
            label_dict = frame_level_label(label_dict)
            self.src.append(wav_data)
            self.trg.append(label_dict)

    def __getitem__(self, index):
        return self.src[index], self.trg[index]

    def __len__(self):
        return len(self.src)


class BatchDataLoader(object):
    def __init__(self, s_mix_dataset, batch_size, is_shuffle=True, workers_num=16):
        self.dataloader = DataLoader(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle,
                                     num_workers=workers_num, collate_fn=self.collate_fn)

    def get_dataloader(self):
        return self.dataloader

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: x[0].size()[0], reverse=True)
        wav, tag = zip(*batch)

        wav_pad = []
        for i, i_data in enumerate(wav):
            wav_pad.append(i_data)

        wav_batch = pad_sequence(wav_pad, batch_first=True)

        tag_batch = np.array(tag).astype(int)
        tag_batch = torch.from_numpy(tag_batch)

        # print(tag)
        return [wav_batch, tag_batch]
        # return [wav, tag_batch]


if __name__ == '__main__':
    data_train = Dataset()
    tr_batch_dataloader = BatchDataLoader(data_train, 32, is_shuffle=True, workers_num=8)
    for i_batch, batch_data in enumerate(tr_batch_dataloader.get_dataloader()):
        print(i_batch)  # 打印batch编号
        # print(batch_data[0])  # 打印该batch里面src （会死掉的）
        print(batch_data[0].shape)
        print(batch_data[1])  # 打印该batch里面trg


