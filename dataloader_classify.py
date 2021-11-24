from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import *
import torch
from tqdm import tqdm
import soundfile as sf
from torch.autograd.variable import *

import numpy as np
import pickle, os


def frame_level_label(label_dict):
    frame_level_label = []
    label_dict = np.pad(label_dict, (200, 200), 'constant', constant_values=(0,0))

    counts = np.bincount(label_dict[200: 400])
    frame_level_label.append(np.argmax(counts))

    index = 200
    for i in range(1, int(label_dict.size / 200) - 2):
        counts = np.bincount(label_dict[index: index+400])
        index += 200
        frame_level_label.append(np.argmax(counts))

    counts = np.bincount(label_dict[index: index+200])
    frame_level_label.append(np.argmax(counts))

    return np.array(frame_level_label)


class Dataset(Dataset):
    def __init__(self, mode='train'):
        super().__init__()
        self.lst=np.load("./"+mode+".npy",allow_pickle=True)
        self.mode=mode
    def __getitem__(self, index):
        wav,_=sf.read(str(self.lst[index]))
        alpha_pow = 1 / (
                np.sqrt(np.sum(wav** 2)) / (wav.size) + 1e-7)
        wav=wav*alpha_pow
        if self.mode=='train':
            label=np.load('/data01/spj/asr_dataset/ai_shell4_vad/TRAIN/seg_label/'+self.lst[index].stem+'.npy')
        else:
            label=np.load('/data01/spj/asr_dataset/ai_shell4_vad/TEST/seg_label/'+self.lst[index].stem+'.npy')
        label=np.minimum(label, 2)
        label=frame_level_label(label)
        # for p, q in tqdm(pack):
        #     wav_data = read_audio(p)
        #
        #     wav_data *= alpha_pow
        #
        #
        #     label_dict = np.load(q)
        #     label_dict = np.minimum(label_dict, 2)
        #     label_dict = frame_level_label(label_dict)
        #     self.src.append(wav_data)
        #     self.trg.append(label_dict)

        # with open(path, 'wb') as f:
        #     pickle.dump((self.src, self.trg), f, pickle.HIGHEST_PROTOCOL)

        sample=(
            Variable(torch.FloatTensor(wav.astype('float32'))),
            Variable(torch.FloatTensor(label.astype('float32'))),
            alpha_pow
        )
        return sample

    def __len__(self):
        return len(self.lst)


class BatchDataLoader(object):
    def __init__(self, s_mix_dataset, batch_size, is_shuffle=True, workers_num=16):
        self.dataloader = DataLoader(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle,
                                     num_workers=workers_num, collate_fn=self.collate_fn)

    def get_dataloader(self):
        return self.dataloader

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: x[0].size()[0], reverse=True)
        wav, tag,alpha_pow = zip(*batch)

        # wav_pad = []
        # for i, i_data in enumerate(wav):
        #     wav_pad.append(i_data)

        wav_batch = pad_sequence(wav, batch_first=True)
        tag_batch=pad_sequence(tag,batch_first=True)


        # print(tag)
        return [wav_batch, tag_batch,alpha_pow]
        # return [wav, tag_batch]


if __name__ == '__main__':
    data_train = Dataset()
    tr_batch_dataloader = BatchDataLoader(data_train, 32, is_shuffle=True, workers_num=8)
    for i_batch, batch_data in enumerate(tr_batch_dataloader.get_dataloader()):
        print(i_batch)  # 打印batch编号
        # print(batch_data[0])  # 打印该batch里面src （会死掉的）
        print(batch_data[0].shape)
        print(batch_data[1])  # 打印该batch里面trg


