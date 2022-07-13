import torch
import torch.nn as nn
import numpy as np
from RNN_model import M_RNN
from BPE_handle.word22id import word2id, id2word, get_word, replace_word

PATH_MODEL = "./RNN_train_state/train_state_73000.pth"
PATH_INDEX = "F:/PyTorch学习/BPE_handle/en_index.npy"
# PATH_INDEX = "./RNN_file/en_index.npy"


def load(srcf):
    v = np.load(srcf, allow_pickle=True).item()  # 字典的导入
    return v


if __name__ == '__main__':
    en2index = load(PATH_INDEX)
    # print(en2index)
    en = get_word(en2index)
    # print(en)

    sentence = input("请输入测试语句:").strip().split()
    # print(sentence)

    sentence = replace_word(sentence, en)
    print(sentence)

    word2id_l = word2id(en2index, sentence)  # !!!
    # print(word2id)

    model = M_RNN(len(en2index))

    net_state_dict = torch.load(PATH_MODEL, map_location='cpu')
    model.load_state_dict(net_state_dict)

    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # pretrain_dict = torch.load(PATH_MODEL)
    # net_state_dict = model.state_dict()
    # pretrain_dict_1 = {k: v for k, v in pretrain_dict.items() if k in net_state_dict}
    # net_state_dict.update(pretrain_dict_1)
    # model.load_state_dict(net_state_dict)

    decode_step = len(sentence)
    predict_step = len(word2id_l)

    state = None
    # print(state.is_cuda)
    result = []

    # 先将整句话过一遍decode
    for i in range(decode_step):
        words = torch.LongTensor([word2id_l[i]]).to(device)
        output, state = model.decode(words, state)
        # print(state)
        # print("**"*40)

    # print(state)

    for i in range(predict_step):
        # print(state)
        words = torch.LongTensor([word2id_l[i]]).to(device)  # 每次取一个词的索引
        output, state = model.decode(words, state)  # 解码获得最可能的下一个词
        result.append(output.item())

    result_words = id2word(en2index, result)
    print(result_words)

    print("预测句子结果为:{}".format(" ".join(result_words).replace("@@ ", "")))
