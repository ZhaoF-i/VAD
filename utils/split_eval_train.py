import random

import path
import numpy as np
import os
from pathlib import Path
if __name__ == '__main__':
    # name_lst=list(Path("/data01/spj/ai_shell4_vad/TRAIN/seg_wav/").rglob('*.wav'))
    name_lst=list(Path("/data01/zhaofei/data/asr_dataset/ai_shell4_vad/TRAIN/seg_wav/").rglob('*.wav'))
    random.shuffle(name_lst)
    len_lst=len(name_lst)
    train_lst=name_lst[:int(0.9*len_lst)]
    eval_lst=name_lst[int(0.9*len_lst):]
    np.save("../validate.npy",eval_lst)
    np.save("../train.npy",train_lst)
    print("列表已经生成")