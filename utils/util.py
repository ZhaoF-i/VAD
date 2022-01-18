import numpy as np
import os
import re
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def one_hot(input, n_class):
    one_hot = torch.zeros(len(input), n_class)
    label_one_hot = one_hot.scatter(1, input, 1)
    return label_one_hot

def frame_level_label(label_dict, frame_len, frame_shift):
    label_dict = np.pad(label_dict, (frame_shift, frame_shift), 'symmetric')
    input = torch.Tensor(label_dict).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones(frame_len).unsqueeze(0).unsqueeze(0)
    torch_label = F.conv1d(input=input, weight=kernel, stride=frame_shift)
    torch_label = torch_label.squeeze()
    half = torch.ones(len(torch_label)) / 2  # + 0.00001
    torch_label /= frame_len

    torch_frame_label = torch_label + half
    torch_out = torch.as_tensor(torch_frame_label, dtype=torch.int64)

    return torch_out

def expandWindow(data, left, right):
    data = data.detach().cpu().numpy()
    sp = data.shape
    idx = 0
    exdata = np.zeros([sp[0], sp[1], sp[2] * (left + right + 1)])
    for i in range(-left, right+1):
        exdata[:, :, sp[2] * idx : sp[2] * (idx + 1)] = np.roll(data, shift=-i, axis=1)
        idx = idx + 1
    return Variable(torch.FloatTensor(exdata)).cuda(CUDA_ID[0])

def context_window(data, left, right):
    sp = data.data.shape
    exdata = torch.zeros(sp[0], sp[1], sp[2] * (left + right + 1)).cuda(CUDA_ID[0])
    for i in range(1, left + 1):
        exdata[:, i:, sp[2] * (left - i) : sp[2] * (left - i + 1)] = data.data[:, :-i,:]
    for i in range(1, right+1):
        exdata[:, :-i, sp[2] * (left + i):sp[2]*(left+i+1)] = data.data[:, i:, :]
    exdata[:, :, sp[2] * left : sp[2] * (left + 1)] = data.data
    return Variable(exdata)
def read_list(list_file):
    f = open(list_file, "r")
    lines = f.readlines()
    list_sig = []
    for x in lines:
        list_sig.append(x[:-1])
    f.close()
    return list_sig
def gen_list(wav_dir, append):
    l = []
    lst = os.listdir(wav_dir)
    lst.sort()
    for f in lst:
        if re.search(append, f):
            l.append(f)
    return l



def write_log(file,name, train, validate):
    message = ''
    for m, val in enumerate(train):
        message += ' --TRerror%i=%.3f ' % (m, val.data.numpy())
    for m, val in enumerate(validate):
        message += ' --CVerror%i=%.3f ' % (m, val.data.numpy())
    file.write(name + ' ')
    file.write(message)
    file.write('/n')

def makedirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def saveYAML(yaml,save_path):
    f_params = open(save_path, 'w')
    for k, v in yaml.items():
        f_params.write('{}:\t{}\n'.format(k, v))