
import torch
from transformers import RobertaModel, RobertaTokenizer





DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')#

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalInformationBottleneck(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalInformationBottleneck, self).__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        self.fc_latent = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.fc_latent(z)
        return x_reconstructed, mu, logvar


class FusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super(FusionModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        # LSTM layer to process sentence embeddings
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Linear layer for classification
        self.linear = nn.Linear(hidden_dim, num_labels)

        # Attention weights for forward and backward fusion
        self.forward_attention = nn.Parameter(torch.full((1, hidden_dim),0.001))
        self.backward_attention = nn.Parameter(torch.full((1, hidden_dim),0.001))

    def forward(self, x, labels=None, is_training=True):
        batch_size, doc_len, _ = x.size()

        # LSTM encoding for each sentence in the document
        lstm_out, _ = self.lstm(x)  # (batch_size, doc_len, hidden_dim)


        forward_fusion = []
        for i in range(doc_len):
            forward_sum = torch.zeros_like(lstm_out[:, 0, :])  # Initialize forward sum
            for j in range(i):
                forward_sum += self.linear(lstm_out[:, j, :]).argmax(dim=1).unsqueeze(1) * lstm_out[:, j,
                                                                                               :] * self.forward_attention
            fused_forward = lstm_out[:, i, :] + forward_sum  # Add the forward fusion effect
            forward_fusion.append(fused_forward)

        forward_fusion = torch.stack(forward_fusion, dim=1)  # Shape: (batch_size, doc_len, hidden_dim)

        backward_fusion = []
        for i in range(doc_len):
            backward_sum = torch.zeros_like(lstm_out[:, 0, :])  # Initialize backward sum
            for j in range(i + 1, doc_len):
                backward_sum += self.linear(lstm_out[:, j, :]).argmax(dim=1).unsqueeze(1) * lstm_out[:, j,
                                                                                                :] * self.backward_attention
            fused_backward = lstm_out[:, i, :] + backward_sum  # Add the backward fusion effect
            backward_fusion.append(fused_backward)

        backward_fusion = torch.stack(backward_fusion, dim=1)  # Shape: (batch_size, doc_len, hidden_dim)

        fused_output = forward_fusion + backward_fusion  # Element-wise sum

        return fused_output



class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.roberta_encoder = RobertaEncoder(configs)

        self.trans_doc = clause_transformer(configs)

        self.pred_emo = Pre_emo_Predictions(configs)
        self.pre=Pre_Predictions(configs)

        self.fc_mu = nn.Linear(1024, 1024)
        self.fc_logvar = nn.Linear(1024, 1024)

        self.fusion=FusionModel(configs.feat_dim*2,512,2).to(DEVICE)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, query, query_mask, query_seg, query_len, seq_len, doc_len, adj,query_ans, q_type):
        # shape: batch_size, max_doc_len, 1024
        doc_sents_h, query_h = self.roberta_encoder(query, query_mask, query_seg, query_len, seq_len, doc_len,q_type)

        doc_sents_h = self.trans_doc(doc_sents_h, doc_len)


        doc_sents_h_final = torch.cat((query_h, doc_sents_h), dim=-1)
        fuse_output=self.fusion(doc_sents_h_final,labels=None)

        pred = self.pred_emo(fuse_output)

        mu = self.fc_mu(doc_sents_h)
        logvar = self.fc_logvar(doc_sents_h)
        z = self.reparameterize(mu, logvar)
        classification = self.pre(z)


        return pred, classification, mu, logvar

    def loss_pre(self, pred, true, mask):
        true = torch.FloatTensor(true.float()).to(DEVICE)  # shape: batch_size, seq_len
        mask = torch.BoolTensor(mask.bool()).to(DEVICE)
        pred = pred.masked_select(mask)
        true = true.masked_select(mask)
        criterion = nn.BCELoss()
        return criterion(pred, true)



class RobertaEncoder(nn.Module):
    def __init__(self, configs):
        super(RobertaEncoder, self).__init__()
        hidden_size = configs.feat_dim
        self.roberta = RobertaModel.from_pretrained(configs.bert_cache_path, output_hidden_states=True)
        self.tokenizer = RobertaTokenizer.from_pretrained(configs.bert_cache_path)
        self.fc = nn.Linear(hidden_size, 1)
        self.fc_query = nn.Linear(hidden_size, 1)

        self.f_emo=torch.nn.Parameter(torch.FloatTensor([0.2,1]))
        self.f_cau=torch.nn.Parameter(torch.FloatTensor([0.2,1]))


    def forward(self, query, query_mask, query_seg, query_len, seq_len, doc_len,q_type):
        outputs = self.roberta(input_ids=query.to(DEVICE),
                                  attention_mask=query_mask.to(DEVICE))
        last_hidden_state = outputs.last_hidden_state
        hidden_states, mask_doc, query_state, mask_query = self.get_sentence_state(last_hidden_state, query_len, seq_len, doc_len,q_type)


        alpha_q = self.fc_query(query_state).squeeze(-1)  # bs, query_len
        mask_query = 1 - mask_query  # bs, max_query_len
        alpha_q.data.masked_fill_(mask_query.bool(), -9e5)
        alpha_q = F.softmax(alpha_q, dim=-1).unsqueeze(-1).repeat(1, 1, query_state.size(-1))
        query_state = torch.sum(alpha_q * query_state, dim=1)  # bs, 768
        query_state = query_state.unsqueeze(1).repeat(1, hidden_states.size(1), 1)

        alpha=self.fc(hidden_states).squeeze(-1)
        mask_doc = 1 - mask_doc # bs, max_doc_len, max_seq_len
        alpha.data.masked_fill_(mask_doc.bool(), -9e5)
        alpha = F.softmax(alpha, dim=-1).unsqueeze(-1).repeat(1, 1, 1, hidden_states.size(-1))
        hidden_states = torch.sum(alpha * hidden_states, dim=2) # bs, max_doc_len, 768

        return hidden_states.to(DEVICE), query_state.to(DEVICE)

    def get_sentence_state(self, hidden_states, query_lens, seq_lens, doc_len,q_type):
        sentence_state_all = []
        query_state_all = []
        mask_all = []
        mask_query = []
        max_seq_len = 0
        for seq_len in seq_lens:
            for l in seq_len:
                max_seq_len = max(max_seq_len, l)
        max_doc_len = max(doc_len)
        max_query_len = max(query_lens)
        for i in range(hidden_states.size(0)):
            query = hidden_states[i, 1: query_lens[i] + 1]
            assert query.size(0) == query_lens[i]
            if query_lens[i] < max_query_len:
                query = torch.cat([query, torch.zeros((max_query_len - query_lens[i], query.size(1))).to(DEVICE)], dim=0)
            query_state_all.append(query.unsqueeze(0))
            mask_query.append([1] * query_lens[i] + [0] * (max_query_len -query_lens[i]))

            mask = []
            begin = query_lens[i] + 3 #0,2,2
            sentence_state = []
            for seq_len in seq_lens[i]:
                sentence = hidden_states[i, begin: begin + seq_len]
                if seq_len - sentence.size(0) == 1:
                    seq_len -= 1
                begin += seq_len

                if sentence.size(0) < max_seq_len:
                    sentence = torch.cat([sentence, torch.zeros((max_seq_len - seq_len, sentence.size(-1))).to(DEVICE)],
                                         dim=0)

                sentence_state.append(sentence.unsqueeze(0))
                mask.append([1] * seq_len + [0] * (max_seq_len - seq_len))
            try:
                sentence_state = torch.cat(sentence_state, dim=0).to(DEVICE)
            except:
                print(sentence_state_per.size(0) for sentence_state_per in sentence_state)
                print(max_seq_len)
                print(seq_lens)
            if sentence_state.size(0) < max_doc_len:
                mask.extend([[0] * max_seq_len] * (max_doc_len - sentence_state.size(0)))
                padding = torch.zeros(
                    (max_doc_len - sentence_state.size(0), sentence_state.size(-2), sentence_state.size(-1)))
                sentence_state = torch.cat([sentence_state, padding.to(DEVICE)], dim=0)
            sentence_state_all.append(sentence_state.unsqueeze(0))
            mask_all.append(mask)
        query_state_all = torch.cat(query_state_all, dim=0).to(DEVICE)
        mask_query = torch.tensor(mask_query).to(DEVICE)
        sentence_state_all = torch.cat(sentence_state_all, dim=0).to(DEVICE)
        mask_all = torch.tensor(mask_all).to(DEVICE)
        return sentence_state_all, mask_all, query_state_all, mask_query



class clause_transformer(nn.Module):
    def __init__(self, configs):
        super(clause_transformer, self).__init__()
        hidden_size=configs.feat_dim
        self.encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size,nhead=8)
        self.transformer_encoder=nn.TransformerEncoder(self.encoder_layer,num_layers=6)


    def forward(self, doc_sents_h, doc_len):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len
        mask=[]
        for i in range(batch):
            mask.append([1] * doc_len[i] + [0] * (max_doc_len - doc_len[i] ))
        mask= torch.tensor(mask).to(DEVICE)
        mask_query = 1 - mask  # bs, max_query_len
        #doc_sents_h.data.masked_fill_(mask_query.bool(), -9e5)
        doc_sents_h=torch.transpose(doc_sents_h,0,1)
        output=self.transformer_encoder(src=doc_sents_h,mask=None,src_key_padding_mask=mask_query.bool())
        output = F.dropout(output, 0.1, training=self.training)
        output=torch.transpose(output,0,1)

        return output

class Pre_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions, self).__init__()
        self.feat_dim = 1024 #configs.feat_dim
        self.out_e = nn.Linear(self.feat_dim, 1)

    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h).squeeze(-1)  # bs, max_doc_len, 1
        pred_e = torch.sigmoid(pred_e)
        return pred_e # shape: bs ,max_doc_len

class Pre_emo_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_emo_Predictions, self).__init__()
        self.feat_dim =512 #configs.feat_dim*2
        self.out_e = nn.Linear(self.feat_dim, 1)

    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h).squeeze(-1)  # bs, max_doc_len, 1
        pred_e = torch.sigmoid(pred_e)
        return pred_e # shape: bs ,max_doc_len
