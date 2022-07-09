import torch
import numpy as np
from RNN_model import M_RNN

PATH_MODEL = "./RNN_train_state/train_state_73000.pth"
PATH_INDEX = "F:/PyTorch学习/BPE_handle/en_index.npy"
# PATH_INDEX = "./RNN_file/en_index.npy"


def load(srcf):
    v = np.load(srcf, allow_pickle=True).item()  # 字典的导入
    return v


if __name__ == '__main__':
    en2index = load(PATH_INDEX)
    # print(en2index)

    sentence = input("请输入测试语句:").strip().split()
    print(sentence)

    word2id = [en2index[i] for i in sentence]
    print(word2id)

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

    predict_step = len(word2id)
    input = []

    for i in range(predict_step):
        words = torch.LongTensor([word2id[i]]).to(device)  # 每次取一个词的索引
        output = model.decode(words)  # 解码获得最可能的下一个词
        input.append(output.item())

    result_words = []

    for i in range(len(input)):
        s = [en for en, index in en2index.items() if index == input[i]]
        result_words.append("".join(s))

    print(" ".join(result_words).replace("@@ ", ""))
