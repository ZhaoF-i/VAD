import random
import soundfile as sf
import path
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    # name_lst=list(Path("/data01/spj/ai_shell4_vad/TRAIN/seg_wav/").rglob('*.wav'))
    wav_lst=list(Path("/data01/spj/asr_dataset/ai_shell4_vad/TRAIN/seg_wav/").rglob('*.wav'))
    label_lst = list(Path("/data01/spj/asr_dataset/ai_shell4_vad/TRAIN/seg_label/").rglob('*.npy'))
    wav_lst = sorted(wav_lst)
    label_lst = sorted(label_lst)  #
    save_path =  '/data01/zhaofei/data/asr_dataset/ai_shell4_vad/TRAIN'

    for (i,j) in tqdm(zip(wav_lst, label_lst)):
        wav, _ = sf.read(i)
        label = np.load(j)
        wav_name = str(i.parent)+'/'+ str(i.stem)
        label_name = str(j.parent)+'/'+ str(j.stem)

        for k in range(10):
            seg_wav = wav[k*16000: (k+1)*16000]
            seg_wav_name = save_path + '/seg_wav/' +str(i.stem) + '_' + str(k) + '.wav'
            sf.write(seg_wav_name, seg_wav, samplerate=16000)

            seg_label = label[k*16000: (k+1)*16000]
            seg_label_name = save_path + '/seg_label/' +str(j.stem) + '_' + str(k) + '.npy'
            np.save(seg_label_name, seg_label)



    # wav, _ = sf.read(wav_lst[0])
    # len_lst=len(name_lst)
    # train_lst=name_lst[:int(0.9*len_lst)]
    # eval_lst=name_lst[int(0.9*len_lst):]
    # np.save("../validate.npy",eval_lst)
    # np.save("../train.npy",train_lst)
    # print("列表已经生成")