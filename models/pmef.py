# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder


class PMEF(nn.Module):
    '''
    PMEF
    '''
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.batch_size = opt.batch_size
        self.dropout = nn.Dropout(opt.drop_out)
        
        self.word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.answerer_embs = nn.Embedding(opt.a_num + 5, opt.word_dim)
        self.tag_embs = nn.Embedding(opt.tag_num + 5, opt.word_dim)

        self.q_title_encoder = Encoder(opt.enc_method, opt.word_dim, opt.fea_size*2, opt.fea_size)
        self.q_body_encoder = Encoder(opt.enc_method, opt.word_dim, opt.fea_size*2, opt.fea_size)
        self.h_title_encoder = Encoder(opt.enc_method, opt.word_dim, opt.fea_size*2, opt.fea_size)
        self.h_body_encoder = Encoder(opt.enc_method, opt.word_dim, opt.fea_size*2, opt.fea_size)

        self.f_fc = nn.Linear(opt.fea_size*2, opt.fea_size)
        self.predict = nn.Linear(opt.fea_size, 1)

        self.q_title_fc = nn.Linear(opt.fea_size, opt.fea_size)
        self.q_body_fc = nn.Linear(opt.fea_size, opt.fea_size)
        self.q_fixFea_qv = nn.Linear(opt.fea_size, opt.fea_size)

        self.q_tag_fc = nn.Linear(opt.fea_size, opt.fea_size)
        self.aid_q_tags_fc = nn.Linear(opt.fea_size, opt.fea_size)

        self.H_body_qv = nn.Linear(opt.fea_size, opt.fea_size)
        self.H_title_qv = nn.Linear(opt.fea_size, opt.fea_size)
        self.H_fixFea_qv = nn.Linear(opt.fea_size, opt.fea_size)


        
        self.reset_para()

    def forward(self, data):

        a_id, q_title_id, q_title_mask, q_body_id, q_body_mask, q_tag, \
            aid_q_titles_id, aid_q_titles_mask, aid_q_bodys_id, aid_q_bodys_mask, aid_q_tags = data

        q_tag_emb = self.q_tag_fc(self.tag_embs(q_tag)).mean(1).unsqueeze(2)
        H_tag = self.aid_q_tags_fc(self.tag_embs(aid_q_tags)).mean(2)
        answerer_emb = self.answerer_embs(a_id)

        q_title_embs = self.word_embs(q_title_id)
        q_body_embs = self.word_embs(q_body_id)
        u_q_list_title = self.word_embs(aid_q_titles_id)
        u_q_list_body = self.word_embs(aid_q_bodys_id)

        q_title_fea = self.q_title_encoder(q_title_embs)
        q_title_fea = q_title_fea * q_title_mask.unsqueeze(2)
        q_titlefea = self.q_title_fc(q_title_fea.mean(1)).unsqueeze(2)

        q_body_fea = self.q_body_encoder(q_body_embs)
        q_body_fea = q_body_fea * q_body_mask.unsqueeze(2)
        q_bodyfea = self.q_body_fc(q_body_fea.mean(1)).unsqueeze(2)

        b, m, l, d = u_q_list_title.size()
        u_q_list_title = u_q_list_title.view(-1, l, d)
        u_q_fea_title = self.h_title_encoder(u_q_list_title)
        u_q_fea_title = u_q_fea_title.mean(1)
        H_title = u_q_fea_title.view(b, m, -1)

        b, m, l, d = u_q_list_body.size()
        u_q_list_body = u_q_list_body.view(-1, l, d)
        u_q_fea_body = self.h_body_encoder(u_q_list_title)
        u_q_fea_body = u_q_fea_body.mean(1) 
        H_body = u_q_fea_body.view(b, m, -1)

        H_titleWeight = torch.bmm(H_title, q_titlefea)
        H_titleScore = F.softmax(H_titleWeight, 1)
        H_titleFea = H_title * H_titleScore
        H_titleFea = self.dropout(H_titleFea.sum(1))

        H_bodyWeight = torch.bmm(H_body, q_bodyfea)
        H_bodyScore = F.softmax(H_bodyWeight, 1)
        H_bodyFea = H_body * H_bodyScore
        H_bodyFea = self.dropout(H_bodyFea.sum(1))

        H_tagWeight = torch.bmm(H_tag, q_tag_emb)
        H_tagScore = F.softmax(H_tagWeight, 1)
        H_tagFea = H_tag * H_tagScore
        H_tagFea = self.dropout(H_tagFea.sum(1))

        q_fixFea = torch.stack([q_tag_emb, q_titlefea, q_bodyfea], 1).squeeze(3)
        q_fix_qv = self.q_fixFea_qv(answerer_emb).unsqueeze(2)
        q_fixWeight = torch.matmul(q_fixFea, q_fix_qv)
        q_fixScore = F.softmax(q_fixWeight, 1)
        q_fixFea = q_fixFea * q_fixScore
        q_fixFea = self.dropout(q_fixFea.sum(1))

        H_fixFea = torch.stack([H_tagFea, H_titleFea, H_bodyFea], 1)
        H_fix_qv = self.H_fixFea_qv(answerer_emb).unsqueeze(2)
        H_fixWeight = torch.bmm(H_fixFea, H_fix_qv)
        H_fixScore = F.softmax(H_fixWeight, 1)
        H_fixFea = H_fixFea * H_fixScore
        H_fixFea = self.dropout(H_fixFea.sum(1))

        f = self.f_fc(torch.cat([q_fixFea, H_fixFea], 1))
        out = self.predict(F.relu(f))

        return out

    def reset_para(self):
        nn.init.uniform_(self.word_embs.weight, -0.2, 0.2)
        nn.init.uniform_(self.answerer_embs.weight, -0.2, 0.2)
        nn.init.uniform_(self.tag_embs.weight, -0.2, 0.2)

        fcs = [self.f_fc, self.predict, self.q_title_fc, self.q_body_fc, self.H_fixFea_qv, \
                self.H_title_qv, self.H_body_qv, self.q_fixFea_qv, self.q_tag_fc, self.aid_q_tags_fc]
        for fc in fcs:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.uniform_(fc.bias, 0.01)
