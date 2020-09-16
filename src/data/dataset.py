#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
# random.seed(0)
# np.random.seed(0)
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from collections import OrderedDict

from nnet.transformers_word_handle import transformers_word_handle
from utils.adj_utils import preprocess_adj, sparse_mxs_to_torch_sparse_tensor


class DocRelationDataset:

    def __init__(self, loader, data_type, params, mappings):
        self.unk_w_prob = params['unk_w_prob']
        self.mappings = mappings
        self.loader = loader
        self.data_type = data_type
        self.data = []
        self.lowercase = params['lowercase']
        self.prune_recall = {"0-max":0, "0-1":0, "0-3":0, "1-3":0, "1-max":0, "3-max":0}
        if 'bert-large' in params['pretrain_l_m'] and 'albert' not in params['pretrain_l_m']:
            self.bert = transformers_word_handle("bert", 'bert-large-uncased-whole-word-masking', dataset=params['dataset'])
        elif 'bert-base' in params['pretrain_l_m'] and 'albert' not in params['pretrain_l_m']:
            self.bert = transformers_word_handle("bert", 'bert-base-uncased', dataset=params['dataset'])
        elif 'albert' in params['pretrain_l_m']:
            self.bert = transformers_word_handle('albert', params['pretrain_l_m'], dataset=params['dataset'])
        elif 'xlnet' in params['pretrain_l_m']:
            self.bert = transformers_word_handle('xlnet', params['pretrain_l_m'], dataset=params['dataset'])
        else:
            print('bert init error')
            exit(0)

    def __len__(self):
        return len(self.data)

    def __call__(self):
        pbar = tqdm(self.loader.documents.keys())
        max_node_cnt = 0
        miss_word = 0
        miss_word_dev = 0
        for pmid in pbar:
            pbar.set_description('  Preparing {} data - PMID {}'.format(self.data_type.upper(), pmid))

            # TEXT
            doc = []
            sens_len = []
            words = []
            start_idx = 0
            for i, sentence in enumerate(self.loader.documents[pmid]):
                words += sentence
                start_idx += len(sentence)
                sent = []
                if self.data_type == 'train':
                    for w, word in enumerate(sentence):
                        if self.lowercase:
                            word = word.lower()
                        if word not in self.mappings.word2index:
                            miss_word += 1
                            sent += [self.mappings.word2index['UNK']]  # UNK words = singletons for train
                        elif (word in self.mappings.singletons) and (random.uniform(0, 1) < float(self.unk_w_prob)):
                            sent += [self.mappings.word2index['UNK']]
                        else:
                            sent += [self.mappings.word2index[word]]

                else:
                    for w, word in enumerate(sentence):
                        if self.lowercase:
                            word = word.lower()
                        if word in self.mappings.word2index:
                            sent += [self.mappings.word2index[word]]
                        else:
                            miss_word_dev += 1
                            sent += [self.mappings.word2index['UNK']]
                assert len(sentence) == len(sent), '{}, {}'.format(len(sentence), len(sent))
                doc += [sent]
                sens_len.append(len(sent))

            bert_token, bert_mask, bert_starts = self.bert.subword_tokenize_to_ids(words)

            # NER
            ner = [0] * sum(sens_len)
            for id_, (e, i) in enumerate(self.loader.entities[pmid].items()):
                for sent_id, m1, m2, itype in zip(i.sentNo.split(':'), i.mstart.split(':'), i.mend.split(':'), i.type.split(':')):
                    for j in range(int(m1), int(m2)):
                        ner[j] = self.mappings.type2index[itype]

            # ENTITIES [id, type, start, end] + NODES [id, type, start, end, node_type_id]
            nodes = []
            ent = []
            ent_sen_mask = np.zeros((len(self.loader.entities[pmid].items()), len(sens_len)), dtype=np.float32)
            for id_, (e, i) in enumerate(self.loader.entities[pmid].items()):
                nodes += [[id_, self.mappings.type2index[i.type.split(':')[0]], min([int(ms) for ms in i.mstart.split(':')]),
                           min([int(me) for me in i.mend.split(':')]), int(i.sentNo.split(':')[0]), 0]]

                for sen_id in i.sentNo.split(':'):
                    ent_sen_mask[id_][int(sen_id)] = 1.0
            entity_size = len(nodes)

            nodes_mention = []
            for id_, (e, i) in enumerate(self.loader.entities[pmid].items()):
                for sent_id, m1, m2 in zip(i.sentNo.split(':'), i.mstart.split(':'), i.mend.split(':')):
                    # ent += [[id_, self.mappings.type2index[i.type.split(':')[0]], min(int(m1), bert_max_len-2), min(int(m2), bert_max_len-1), int(sent_id), 1]]
                    ent += [[id_, self.mappings.type2index[i.type.split(':')[0]], int(m1), int(m2), int(sent_id), 1]]
                    nodes_mention += [[id_, self.mappings.type2index[i.type.split(':')[0]], int(m1), int(m2), int(sent_id), 1]]

            ent.sort(key=lambda x: x[0], reverse=False)
            nodes_mention.sort(key=lambda x: x[0], reverse=False)
            nodes += nodes_mention

            for s, sentence in enumerate(self.loader.documents[pmid]):
                nodes += [[s, s, s, s, s, 2]]

            nodes = np.array(nodes)

            max_node_cnt = max(max_node_cnt, nodes.shape[0])
            ent = np.array(ent)

            # RELATIONS
            ents_keys = list(self.loader.entities[pmid].keys())  # in order
            trel = -1 * np.ones((len(ents_keys), len(ents_keys)))
            relation_multi_label = np.zeros((len(ents_keys), len(ents_keys), self.mappings.n_rel))
            rel_info = np.empty((len(ents_keys), len(ents_keys)), dtype='object_')
            for id_, (r, ii) in enumerate(self.loader.pairs[pmid].items()):
                rt = np.random.randint(len(ii))
                trel[ents_keys.index(r[0]), ents_keys.index(r[1])] = self.mappings.rel2index[ii[0].type]
                relation_set = set()
                for i in ii:
                    assert relation_multi_label[ents_keys.index(r[0]), ents_keys.index(r[1]), self.mappings.rel2index[i.type]] != 1.0
                    relation_multi_label[ents_keys.index(r[0]), ents_keys.index(r[1]), self.mappings.rel2index[i.type]] = 1.0
                    assert self.loader.ign_label == "NA" or self.loader.ign_label == "1:NR:2"
                    if i.type != self.loader.ign_label:
                        assert relation_multi_label[ents_keys.index(r[0]), ents_keys.index(r[1]), self.mappings.rel2index[self.loader.ign_label]] != 1.0
                    relation_set.add(self.mappings.rel2index[i.type])

                    if i.type != self.loader.ign_label:
                        dis_cross = int(i.cross)
                        if dis_cross == 0:
                            self.prune_recall['0-1'] += 1
                            self.prune_recall['0-3'] += 1
                            self.prune_recall['0-max'] += 1
                        elif dis_cross < 3:
                            self.prune_recall['0-3'] += 1
                            self.prune_recall['1-3'] += 1
                            self.prune_recall['1-max'] += 1
                            self.prune_recall['0-max'] += 1
                        else:
                            self.prune_recall['0-max'] += 1
                            self.prune_recall['3-max'] += 1
                            self.prune_recall['1-max'] += 1

                rel_info[ents_keys.index(r[0]), ents_keys.index(r[1])] = OrderedDict(
                                                                            [('pmid', pmid),
                                                                            ('sentA', self.loader.entities[pmid][r[0]].sentNo),
                                                                            ('sentB',
                                                                            self.loader.entities[pmid][r[1]].sentNo),
                                                                            ('doc', self.loader.documents[pmid]),
                                                                            ('entA', self.loader.entities[pmid][r[0]]),
                                                                            ('entB', self.loader.entities[pmid][r[1]]),
                                                                            ('rel', relation_set),
                                                                            ('dir', ii[rt].direction), ('intrain', ii[rt].intrain),
                                                                            ('cross', ii[rt].cross)])

                assert nodes[ents_keys.index(r[0])][2] == min([int(ms) for ms in self.loader.entities[pmid][r[0]].mstart.split(':')])

            #######################
            # DISTANCES
            #######################
            xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')

            r_id, c_id = nodes[xv, 5], nodes[yv, 5]  # node type
            r_Eid, c_Eid = nodes[xv, 0], nodes[yv, 0]
            r_Sid, c_Sid = nodes[xv, 4], nodes[yv, 4]
            r_Ms, c_Ms = nodes[xv, 2], nodes[yv, 2]
            r_Me, c_Me = nodes[xv, 3]-1, nodes[yv, 3]-1

            # dist feature
            dist_dir_h_t = np.full((r_id.shape[0], r_id.shape[0]), 0)

            # MM: mention-mention
            a_start = np.where(np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3), r_Ms, -1)
            b_start = np.where(np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3), c_Ms, -1)

            dis = a_start - b_start
            dis_index = np.where(dis < 0, -self.mappings.dis2idx_dir[-dis], self.mappings.dis2idx_dir[dis])
            condition = (np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3)
                                 & (a_start != -1) & (b_start != -1))
            dist_dir_h_t = np.where(condition, dis_index, dist_dir_h_t)

            # EE: entity-entity
            a_start = np.where((r_id == 0) & (c_id == 0), r_Ms, -1)
            b_start = np.where((r_id == 0) & (c_id == 0), c_Ms, -1)
            dis = a_start - b_start

            dis_index = np.where(dis < 0, -self.mappings.dis2idx_dir[-dis], self.mappings.dis2idx_dir[dis])
            condition = ((r_id == 0) & (c_id == 0) & (a_start != -1) & (b_start != -1))
            dist_dir_h_t = np.where(condition, dis_index, dist_dir_h_t)

            #######################
            # GRAPH CONNECTIONS
            #######################
            adjacency = np.full((r_id.shape[0], r_id.shape[0]), 0, 'i')
            rgcn_adjacency = np.full((5, r_id.shape[0], r_id.shape[0]), 0.0)

            # mention-mention
            adjacency = np.where(np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3) & (r_Sid == c_Sid), 1, adjacency)  # in same sentence
            rgcn_adjacency[0] = np.where(
                    np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3) & (r_Sid == c_Sid), 1,
                    rgcn_adjacency[0])

            # entity-mention
            adjacency = np.where((r_id == 0) & (c_id == 1) & (r_Eid == c_Eid), 1, adjacency)  # belongs to entity
            adjacency = np.where((r_id == 1) & (c_id == 0) & (r_Eid == c_Eid), 1, adjacency)
            rgcn_adjacency[1] = np.where((r_id == 0) & (c_id == 1) & (r_Eid == c_Eid), 1, rgcn_adjacency[1])  # belongs to entity
            rgcn_adjacency[1] = np.where((r_id == 1) & (c_id == 0) & (r_Eid == c_Eid), 1, rgcn_adjacency[1])

            # sentence-sentence (direct + indirect)
            adjacency = np.where((r_id == 2) & (c_id == 2), 1, adjacency)
            rgcn_adjacency[2] = np.where((r_id == 2) & (c_id == 2), 1, rgcn_adjacency[2])

            # mention-sentence
            adjacency = np.where(np.logical_or(r_id == 1, r_id == 3) & (c_id == 2) & (r_Sid == c_Sid), 1, adjacency)  # belongs to sentence
            adjacency = np.where((r_id == 2) & np.logical_or(c_id == 1, c_id == 3) & (r_Sid == c_Sid), 1, adjacency)
            rgcn_adjacency[3] = np.where(np.logical_or(r_id == 1, r_id == 3) & (c_id == 2) & (r_Sid == c_Sid), 1, rgcn_adjacency[3])  # belongs to sentence
            rgcn_adjacency[3] = np.where((r_id == 2) & np.logical_or(c_id == 1, c_id == 3) & (r_Sid == c_Sid), 1, rgcn_adjacency[3])

            # entity-sentence
            for x, y in zip(xv.ravel(), yv.ravel()):
                if nodes[x, 5] == 0 and nodes[y, 5] == 2:  # this is an entity-sentence edge
                    z = np.where((r_Eid == nodes[x, 0]) & (r_id == 1) & (c_id == 2) & (c_Sid == nodes[y, 4]))

                    # at least one M in S
                    temp_ = np.where((r_id == 1) & (c_id == 2) & (r_Sid == c_Sid), 1, adjacency)
                    temp_ = np.where((r_id == 2) & (c_id == 1) & (r_Sid == c_Sid), 1, temp_)
                    adjacency[x, y] = 1 if (temp_[z] == 1).any() else 0
                    adjacency[y, x] = 1 if (temp_[z] == 1).any() else 0
                    rgcn_adjacency[4][x, y] = 1 if (temp_[z] == 1).any() else 0
                    rgcn_adjacency[4][y, x] = 1 if (temp_[z] == 1).any() else 0

            rgcn_adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(rgcn_adjacency[i]) for i in range(5)])


            dist_dir_h_t = dist_dir_h_t[0: entity_size, 0:entity_size]
            self.data += [{'ents': ent, 'rels': trel, 'multi_rels': relation_multi_label,
                           'dist_dir': dist_dir_h_t, 'text': doc, 'info': rel_info,
                           'adjacency': adjacency, 'rgcn_adjacency': rgcn_adjacency, 'ners': np.array(ner),
                           'section': np.array([len(self.loader.entities[pmid].items()), ent.shape[0], len(doc), sum([len(s) for s in doc])]),
                           'word_sec': np.array([len(s) for s in doc]),
                           'words': np.hstack([np.array(s) for s in doc]), 'bert_token': bert_token, 'bert_mask': bert_mask, 'bert_starts': bert_starts}]
        print("miss_word", miss_word)
        print("miss_word_dev ", miss_word_dev)
        return self.data, self.prune_recall
