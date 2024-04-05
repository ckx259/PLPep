import os
import argparse
import re, torch
import torch.nn as nn
import time
import esm
import torch.nn.functional as F

class Seq2ESM2andProt(nn.Module):
    def __init__(self, esm2_model_path):
        super(Seq2ESM2andProt, self).__init__()
        self.device = args.device
        self.Esm2, self.alphabet = esm.pretrained.load_model_and_alphabet_local(esm2_model_path)

        self.tokenizer_esm2 = self.alphabet.get_batch_converter()

    def tokenize(self, seq):
        """
        :param tuple_list: e.g., [('seq1', 'FFFFF'), ('seq2', 'AAASDA')]
        """
        tuple_list = [("seq", "{}".format(seq))]
        with torch.no_grad():
            _, _, tokens = self.tokenizer_esm2(tuple_list)
            return tokens.to(self.device)

    def embed(self, seq):
        with torch.no_grad():
            if len(seq) < 5000:
                # [B, L_rec, D]
                token = self.tokenize(seq)
                emb_esm2 = self.Esm2(token, repr_layers=[33])["representations"][33][..., 1:-1, :]
                emb_esm2 = emb_esm2.reshape(-1, emb_esm2.size(-1))
                return emb_esm2
            else:
                embs_esm2 = None
                for ind in range(0, len(seq), 5000):
                    sind = ind
                    eind = min(ind + 5000, len(seq))
                    sub_seq = seq[sind:eind]
                    print(len(sub_seq), len(seq))
                    token, encoded_input = self.tokenize(sub_seq)
                    sub_emb_esm2 = self.Esm2(token, repr_layers=[33])["representations"][33][..., 1:-1, :]
                    sub_emb_esm2 = sub_emb_esm2.reshape(-1, sub_emb_esm2.size(-1))
                    if None is embs_esm2:
                        embs_esm2 = sub_emb_esm2
                    else:
                        embs_esm2 = torch.cat([embs_esm2, sub_emb_esm2], dim=0)
                print(embs_esm2.size())
                return embs_esm2

class Residue_CNN(nn.Module):
    def __init__(self, fea_size, layers):
        super(Residue_CNN, self).__init__()
        self.num_layers = layers
        self.cov_layers = nn.ModuleList()  # 存储卷积层
        for i in range(self.num_layers):
            self.cov_layers.append(nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                # nn.Dropout(0.3),
                nn.Conv1d(64, 1, kernel_size=3, padding=1),
                nn.BatchNorm1d(1),
            ))

        self.avgpool = nn.AdaptiveAvgPool1d(fea_size)


    def forward(self, x):
        x = x.reshape(x.size(0), 1, x.size(-1))
        for cov_layer in self.cov_layers:
            x_c = cov_layer(x)
            x = x + x_c
        x = self.avgpool(x)
        x = x.reshape(-1, x.size(-1))
        return x

class PLPep(nn.Module):
    def __init__(self, layers=1):
        super(PLPep, self).__init__()

        self.rc = Residue_CNN(1280, layers)

        self.mid_linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 256),
        )

        self.classier = nn.Sequential(
            nn.ELU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 2),
        )

    def forward(self, esm2):
        esm2_b1 = self.rc(esm2)
        esm2_b = self.mid_linear(esm2_b1)
        out = self.classier(esm2_b)
        return out, esm2_b

def loadFasta(fasta):
    with open(fasta, 'r') as f:
        lines = f.readlines()
    ans = {}
    name = ''
    seq_list = []
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if 1 < len(name):
                ans[name] = "".join(seq_list)
            name = line[1:]
            seq_list = []
        else:
            seq_list.append(line)
    if 0 < seq_list.__len__():
        ans[name] = "".join(seq_list)
    return ans

def exists(fileOrFolderPath):
    return os.path.exists(fileOrFolderPath)

def createFolder(folder):
    if not exists(folder):
        os.makedirs(folder)

def dateTag():
    time_tuple = time.localtime(time.time())
    yy = time_tuple.tm_year
    mm = "{}".format(time_tuple.tm_mon)
    dd = "{}".format(time_tuple.tm_mday)
    if len(mm) < 2:
        mm = "0" + mm
    if len(dd) < 2:
        dd = "0" + dd

    date_tag = "{}{}{}".format(yy, mm, dd)
    return date_tag

def timeTag():
    time_tuple = time.localtime(time.time())
    hour = "{}".format(time_tuple.tm_hour)
    minuse = "{}".format(time_tuple.tm_min)
    second = "{}".format(time_tuple.tm_sec)
    if len(hour) < 2:
        hour = "0" + hour
    if len(minuse) < 2:
        minuse = "0" + minuse
    if len(second) < 2:
        second = "0" + second

    time_tag = "{}:{}:{}".format(hour, minuse, second)
    return time_tag

def timeRecord(time_log, content):
    date_tag = dateTag()
    time_tag = timeTag()
    with open(time_log, 'a') as file_object:
        file_object.write("{} {} says: {}\n".format(date_tag, time_tag, content))

def parsePredProbs(outs):
    """
    :param outs [Tensor]: [*, 2 or 1]
    :return pred_probs: [*], tgts: [*]
    """

    # 1 : one probability of each sample
    # 2 : two probabilities of each sample
    __type = 1
    if outs.size(-1) == 2:
        __type = 2
        outs = outs.view(-1, 2)
    else:
        outs = outs.view(-1, 1)

    sam_num = outs.size(0)

    outs = outs.tolist()

    pred_probs = []
    for j in range(sam_num):
        out = outs[j]
        if 2 == __type:
            prob_posi = out[1]
            prob_nega = out[0]
        else:
            prob_posi = out[0]
            prob_nega = 1.0 - prob_posi

        sum = prob_posi + prob_nega

        if sum < 1e-99:
            pred_probs.append(0.)
        else:
            pred_probs.append(prob_posi / sum)

    return pred_probs

def getPredLabs(predprobs_list, thre):
    res = []
    for pre in predprobs_list:
        if pre > thre:
            res.append(1)
        else:
            res.append(0)
    return res


if __name__ == '__main__':
    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--savefolder")
    parser.add_argument("-seq_fa", "--seq_fa")
    parser.add_argument("-m", "--model_name")
    parser.add_argument("-l", "--layers", type=int)
    parser.add_argument("-sind", "--start_index", type=int, default=0)
    parser.add_argument("-eind", "--end_index", type=int, default=-1)
    parser.add_argument("-mdn", "--modeldirname", type=str, default='PLPep_model_path')
    parser.add_argument("-sc", "--set_cutoff", type=float, default=-1.0)
    parser.add_argument("-cuda", type=bool, default=True)
    parser.add_argument("-dv", "--device", default='cuda:0')
    args = parser.parse_args()

    if args.savefolder is None or args.seq_fa is None:
        parser.print_help()
        exit("PLEASE INPUT YOUR PARAMETERS CORRECTLY")
    if not (args.set_cutoff == -1 or (args.set_cutoff >= 0.0 and args.set_cutoff <= 1.0)):
        exit("PLEASE INPUT CORRECT CUTOFF")

    savefolder = args.savefolder
    createFolder(savefolder)

    timeRecord("{}/run.time".format(savefolder), "Start")

    seq_fa = args.seq_fa
    esm2m = "{}/pre_model/esm2_t33_650M_UR50D.pt".format(os.path.abspath('.'))
    e2epepmd = "{}/{}/".format(os.path.abspath('.'), args.modeldirname)
    seq_dict = loadFasta(seq_fa)
    start_index = args.start_index
    end_index = args.end_index
    if end_index <= start_index:
        end_index = len(seq_dict)

    keys = []
    for key in seq_dict:
        keys.append(key)

    print('*' * 60 + 'Test Starting' + '*' * 60)
    tot_seq_num = len(seq_dict)


    model = PLPep(args.layers)
    if args.cuda:
        esm2_prot_pre = Seq2ESM2andProt(esm2m)
        esm2_prot_pre.to(args.device)
        checkpoint = torch.load(e2epepmd + os.sep + args.model_name)
        model.to(args.device)
    else:
        esm2_prot_pre = Seq2ESM2andProt(esm2m)
        checkpoint = torch.load(e2epepmd + os.sep + args.model_name, map_location='cpu')
    if args.set_cutoff == -1.0:
        thre = checkpoint['thre']
    else:
        thre = args.set_cutoff
    model.load_state_dict(checkpoint['model'])

    for name, param in esm2_prot_pre.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        param.requires_grad = False

    model.eval()
    int_record = []
    with torch.no_grad():
        for ind in range(tot_seq_num):
            if ind < start_index or ind >= end_index:
                continue
            key = keys[ind]
            seq = seq_dict[key]
            if ind % 1 == 0:
                print("The {}/{}-th {}({}) is predicting...".format(ind, tot_seq_num, key, len(seq)))

            predict_list = []
            single_predict_list = []
            pre_lab_list = []

            filepath = "{}/{}.pred".format(savefolder, key)
            if os.path.exists(filepath):
                continue
            esm2_emb = esm2_prot_pre.embed(seq)
            # print(esm2_emb)
            # print(esm2_emb.size())
            # exit()
            if args.cuda:
                esm2_emb = esm2_emb.cuda()
            out, _ = model(esm2_emb)
            out = F.softmax(out, dim=1)
            predict_list.append(parsePredProbs(out))
            pre_lab_list.append(getPredLabs(parsePredProbs(out), thre))

            with open(filepath, 'w') as file_object:
                # 获取子列表的长度，假设所有子列表长度相同
                list_length = len(pre_lab_list[0])
                file_object.write("Index\tAA\tProb0[cutoff:{}]\tState\n".format(thre))

                # 使用列表解析比较每个索引位置的1和0的数量，并生成新的列表
                result = ['B' if sum(sublist[i] == 1 for sublist in pre_lab_list) > sum(sublist[i] == 0 for sublist in pre_lab_list) else 'N' for i in range(list_length)]

                for i in range(list_length):
                    file_object.write("{}\t{}\t{}\t{}\n".format(i, seq[i], predict_list[0][i], result[i]))
                file_object.close()
    print(int_record)