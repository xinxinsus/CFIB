from config import *
from data_loader import *
from utils.utils import *
from torch.utils.data import Dataset
from networks.network_cfib import Network
import numpy as np
from transformers import RobertaModel, RobertaTokenizer


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')#

if torch.cuda.is_available():
    # 获取CUDA设备数量
    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_gpus}")

    # 获取CUDA设备编号和名称
    for gpu_id in range(num_gpus):
        print(f"CUDA device {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
else:
    print("No CUDA devices available.")



def precision_recall_f1(tp, fp, fn):
    """计算精确率、召回率和F1分数"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def calculate_pos_neg_f1(y_true, y_pred):
    """计算Positive F1 (F1pos) 和 Negative F1 (F1neg)"""
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    TN = ((y_pred == 0) & (y_true == 0)).sum()

    precision_pos, recall_pos, f1_pos = precision_recall_f1(TP, FP, FN)

    precision_neg, recall_neg, f1_neg = precision_recall_f1(TN, FN, FP)

    return precision_pos, recall_pos, f1_pos, precision_neg, recall_neg, f1_neg




class OriginalDataset(Dataset):
    def __init__(self, pre_data):
        self.docid_list = pre_data['_docid_list']
        self.clause_list = pre_data['_clause_list']
        self.doc_len_list = pre_data['_doc_len_list']
        self.clause_len_list = pre_data['_clause_len_list']
        self.pairs = pre_data['_pairs']

        self._f_emo_query = pre_data['_f_emo_query']  # [1, max_for_emo_len]
        self._f_cau_query = pre_data['_f_cau_query']  # [max_for_num, max_for_cau_len]
        self._f_emo_query_len = pre_data['_f_emo_query_len']
        self._f_cau_query_len = pre_data['_f_cau_query_len']
        self._f_emo_query_answer = pre_data['_f_emo_query_answer']
        self._f_cau_query_answer = pre_data['_f_cau_query_answer']
        self._f_emo_query_mask = pre_data['_f_emo_query_mask']  # [1,max_for_emo_len]
        self._f_cau_query_mask = pre_data['_f_cau_query_mask']  # [max_for_num, max_for_cau_len]
        self._f_emo_query_seg = pre_data['_f_emo_query_seg']  # [1,max_for_emo_len]
        self._f_cau_query_seg = pre_data['_f_cau_query_seg']  # [max_for_num, max_for_cau_len]


        self._forward_c_num = pre_data['_forward_c_num']

def organize_data(candidate_index, label, pre):
    new_label = [0] * len(candidate_index)
    new_pre = [0] * len(candidate_index)
    for i, idx in enumerate(candidate_index):
        if idx == -1:
            new_label[i] = 0
            new_pre[i] = 0
        else:
            new_label[i] = label[idx-1]
            new_pre[i] = pre[idx-1]
    return new_label, new_pre


def evaluate_one_batch(configs, batch, model, tokenizer):

    docid_list, clause_list, pairs, \
    feq, feq_mask, feq_seg, feq_len, fe_clause_len, fe_doc_len, fe_adj, feq_an, fe_an_mask, \
    fcq, fcq_mask, fcq_seg, fcq_len, fc_clause_len, fc_doc_len, fc_adj, fcq_an, fc_an_mask,candidate_index_list= batch
    doc_id, clause_list, true_pairs = docid_list[0], clause_list[0], pairs[0]
    candidate_index=candidate_index_list[0]
    true_emo, true_cau = zip(*true_pairs)
    true_emo, true_cau = list(true_emo), list(true_cau)
    temp_text = ' '.join(clause_list)
    text = ' '.join(temp_text.split(' ')).split(' ')

    # step 2
    for idx_emo in set(true_emo):
        f_cau_pred,classi,_,_ = model(fcq, fcq_mask, fcq_seg, fcq_len, fc_clause_len, fc_doc_len, fc_adj,fcq_an, 'f_cau')
        temp_cau_f_prob = f_cau_pred[0].tolist()
        temp_2=classi[0].tolist()
        for idx_cau in range(len(temp_cau_f_prob)):
            if (temp_cau_f_prob[idx_cau] > 0.6 and temp_2[idx_cau]>0.4 ) and abs(idx_emo-1 - idx_cau) <= 10:  #
                        temp_cau_f_prob[idx_cau]=1

        temp_pre = [0 if item != 1 else item for item in temp_cau_f_prob]
        temp_ans= fcq_an[0].tolist()
        new_label, new_pre = organize_data(candidate_index, temp_ans, temp_pre)
    return new_label, new_pre


def evaluate(configs, test_loader, model, tokenizer):
    model.eval()
    pre_list=[]
    label_list=[]
    for batch in test_loader:
        new_label, new_pre = evaluate_one_batch(configs, batch, model, tokenizer)
        pre_list.extend(new_pre)
        label_list.extend(new_label)
    pos_precision, pos_recall, pos_f1,neg_precision, neg_recall, neg_f1 = calculate_pos_neg_f1(label_list, pre_list)

    return pos_precision, pos_recall, pos_f1,neg_precision, neg_recall, neg_f1,(pos_f1+neg_f1)/2


def main(configs, fold_id, tokenizer):
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    data_path = 'data/dm.pt'
    total_data = torch.load(data_path)
    test_loader = build_dataset(configs, total_data['test'], mode='test')

    # model_1
    model = Network(configs).to(DEVICE)
    model.zero_grad()

    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load('model_dia/model_valid.pth'.format(fold_id))['model'])
        pos_precision, pos_recall, pos_f1,neg_precision, neg_recall, neg_f1,macro_f1 = evaluate(configs, test_loader, model, tokenizer)
    return pos_precision, pos_recall, pos_f1,neg_precision, neg_recall, neg_f1,macro_f1


if __name__ == '__main__':
    configs = Config()
    t =  RobertaTokenizer.from_pretrained(configs.bert_cache_path)
    result_e, result_c, result_p = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    res, rcs = [0, 0, 0], [0, 0, 0]
    for fold_id in range(1,2):
        print('===== fold {} ====='.format(fold_id))
        pos_precision, pos_recall, pos_f1,neg_precision, neg_recall, neg_f1,macro_f1 = main(configs, fold_id, t)

    print('Pos- pre:{}, rec:{}, f1:{}'.format(pos_precision, pos_recall, pos_f1))
    print('neg- pre:{}, rec:{}, f1:{}'.format(neg_precision, neg_recall, neg_f1))
    print('macro - f1:{}'.format(macro_f1))



