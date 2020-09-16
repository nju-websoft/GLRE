#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys

import numpy as np
import argparse
import yaml
import yamlordereddictloader
from collections import OrderedDict
from data.reader import read
import os


class ConfigLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_cmd():
        parser = argparse.ArgumentParser()
        # parser.add_argument('--config', type=str, required=True, help='Yaml parameter file')
        parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
        parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')

        parser.add_argument("--feature", default=str)

        parser.add_argument("--remodelfile", type=str)
        parser.add_argument('--input_theta', type=float, default=-1)

        parser.add_argument('--config_file', type=str)
        parser.add_argument('--output_path', type=str, default="dev")
        parser.add_argument('--test_data', type=str)
        parser.add_argument('--save_pred', type=str)

        parser.add_argument('--batch', type=int, help='batch size')
        # parser.add_argument()
        return parser.parse_args()

    def load_config(self):
        inp = self.load_cmd()

        with open(inp.config_file, 'r', encoding="utf-8") as f:
            parameters = yaml.load(f, Loader=yamlordereddictloader.Loader)

        parameters = dict(parameters)
        if not inp.train and not inp.test:
            print('Please specify train/test mode.')
            sys.exit(0)

        parameters['feature'] = inp.feature
        parameters['train'] = inp.train
        parameters['test'] = inp.test
        parameters['config'] = inp.config_file
        parameters['remodelfile'] = inp.remodelfile
        parameters['input_theta'] = inp.input_theta
        parameters['output_path'] = inp.output_path
        if inp.test_data:
            parameters['test_data'] = inp.test_data
        if inp.batch:
            parameters['batch'] = inp.batch
        if inp.save_pred:
            parameters['save_pred'] = inp.save_pred

        return parameters


class DataLoader:
    def __init__(self, input_file, params, trainLoader=None):
        self.input = input_file
        self.params = params

        self.pre_words = []
        self.pre_embeds = OrderedDict()
        self.max_distance = -9999999999
        self.singletons = []
        self.label2ignore = -1
        self.ign_label = self.params['label2ignore']
        self.dataset = params['dataset']
        if params['dataset'] == "docred":
            self.base_file = "./data/DocRED/processed/"
        else:
            self.base_file = "./data/CDR/processed/"

        self.entities_cor_id = None
        if self.params['emb_method']:
            self.word2index = json.load(open(os.path.join(self.base_file, "word2id.json")))
        else:
            self.word2index = json.load(open(os.path.join(self.base_file, "word2id_vec.json")))
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.n_words, self.word2count = len(self.word2index.keys()), {'<UNK>': 1}

        self.type2index = json.load(open(os.path.join(self.base_file, 'ner2id.json')))
        self.index2type = {v: k for k, v in self.type2index.items()}
        self.n_type, self.type2count = len(self.type2index.keys()), {}

        self.rel2index = json.load(open(os.path.join(self.base_file, 'rel2id.json')))
        self.index2rel = {v: k for k, v in self.rel2index.items()}
        self.n_rel, self.rel2count = len(self.rel2index.keys()), {}

        self.documents, self.entities, self.pairs, self.pronouns_mentions = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

        self.dis2idx_dir = np.zeros((800), dtype='int64') # distance feature
        self.dis2idx_dir[1] = 1
        self.dis2idx_dir[2:] = 2
        self.dis2idx_dir[4:] = 3
        self.dis2idx_dir[8:] = 4
        self.dis2idx_dir[16:] = 5
        self.dis2idx_dir[32:] = 6
        self.dis2idx_dir[64:] = 7
        self.dis2idx_dir[128:] = 8
        self.dis2idx_dir[256:] = 9
        self.dis_size = 20


    def find_ignore_label(self):
        """
        Find relation Id to ignore
        """
        print("index2rel\t", self.index2rel)
        for key, val in self.index2rel.items():
            if val == self.ign_label:
                self.label2ignore = key
        assert self.label2ignore != -1
        print("label2ignore ", self.label2ignore)

    @staticmethod
    def check_nested(p):
        starts1 = list(map(int, p[8].split(':')))
        ends1 = list(map(int, p[9].split(':')))

        starts2 = list(map(int, p[14].split(':')))
        ends2 = list(map(int, p[15].split(':')))

        for s1, e1, s2, e2 in zip(starts1, ends1, starts2, ends2):
            if bool(set(np.arange(s1, e1)) & set(np.arange(s2, e2))):
                print('nested pair', p)

    def find_singletons(self, min_w_freq=1):
        """
        Find items with frequency <= 2 and based on probability
        """
        self.singletons = frozenset([elem for elem, val in self.word2count.items()
                                     if (val <= min_w_freq) and elem != 'UNK'])

    def add_relation(self, rel):
        assert rel in self.rel2index
        if rel not in self.rel2index:
            self.rel2index[rel] = self.n_rel
            self.rel2count[rel] = 1
            self.index2rel[self.n_rel] = rel
            self.n_rel += 1
        else:
            if rel not in self.rel2count:
                self.rel2count[rel] = 0
            self.rel2count[rel] += 1

    def add_word(self, word):
        if self.params['lowercase']:
            word = word.lower()
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            if word not in self.word2count:
                self.word2count[word] = 0
            self.word2count[word] += 1

    def add_type(self, type):
        if type not in self.type2index:
            self.type2index[type] = self.n_type
            self.type2count[type] = 1
            self.index2type[self.n_type] = type
            self.n_type += 1
        else:
            if type not in self.type2count:
                self.type2count[type] = 0
            self.type2count[type] += 1

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_document(self, document):
        for sentence in document:
            self.add_sentence(sentence)

    def load_embeds(self, word_dim):
        """
        Load pre-trained word embeddings if specified
        """
        self.pre_embeds = OrderedDict()
        with open(self.params['embeds'], 'r', encoding='utf-8') as vectors:
            for x, line in enumerate(vectors):

                if x == 0 and len(line.split()) == 2:
                    words, num = map(int, line.rstrip().split())
                else:
                    word = line.rstrip().split()[0]
                    vec = line.rstrip().split()[1:]

                    n = len(vec)
                    if n != word_dim:
                        print('Wrong dimensionality! -- line No{}, word: {}, len {}'.format(x, line.rstrip(), n))
                        continue
                    self.add_word(word)
                    self.pre_embeds[word] = np.asarray(vec, 'f')
        self.pre_words = [w for w, e in self.pre_embeds.items()]
        print('  Found pre-trained word embeddings: {} x {}'.format(len(self.pre_embeds), word_dim), end="\n")

    def load_doc_embeds(self):
        self.pre_embeds = OrderedDict()
        word2id = json.load(open('./data/DocRED/processed/word2id_vec.json', 'r', encoding='utf-8'))
        id2word = {id: word for word, id in word2id.items()}
        import numpy as np
        vecs = np.load('./data/DocRED/processed/vec.npy')
        word_dim = 100
        for id in range(vecs.shape[0]):
            word = id2word.get(id)
            vec = vecs[id]
            word_dim = vec.shape
            self.add_word(word)
            self.pre_embeds[word] = np.asarray(vec)
            if self.params['lowercase']:
                self.pre_embeds[word.lower()] = np.asarray(vec)
        self.pre_words = [w for w, e in self.pre_embeds.items()]
        print('  Found pre-trained word embeddings: {} x {}'.format(len(self.pre_embeds), word_dim), end="\n")

    def find_max_length(self, length):
        """ Maximum distance between words """
        for l in length:
            if l-1 > self.max_distance:
                self.max_distance = l-1

    def read_n_map(self):
        """
        Read input.
        """
        lengths, sents, self.documents, self.entities, self.pairs, self.entities_cor_id = \
            read(self.input, self.documents, self.entities, self.pairs)

        self.find_max_length(lengths)

        # map types and positions and relation types
        for did, d in self.documents.items():
            self.add_document(d)

        for did, p in self.pairs.items():
            for k, vs in p.items():
                for v in vs:
                    self.add_relation(v.type)


    def statistics(self):
        """
        Print statistics for the dataset
        """
        print('  Documents: {:<5}\n  Words: {:<5}'.format(len(self.documents), self.n_words))

        print('  Relations: {}'.format(sum([v for k, v in self.rel2count.items()])))
        for k, v in sorted(self.rel2count.items()):
            print('\t{:<10}\t{:<5}\tID: {}'.format(k, v, self.rel2index[k]))

        print('  Max entities number in document: {}'.format(max([len(e) for e in self.entities.values()])))  #41(train)  42(dev)
        print('  Entities: {}'.format(sum([len(e) for e in self.entities.values()])))
        for k, v in sorted(self.type2count.items()):
            print('\t{:<10}\t{:<5}\tID: {}'.format(k, v, self.type2index[k]))

        # 28(train) 27(dev)
        print('  Singletons: {}/{}'.format(len(self.singletons), self.n_words))



    def __call__(self, embeds=None, parameters=None):
        self.read_n_map()
        self.find_ignore_label()
        self.find_singletons(self.params['min_w_freq'])  # words with freq=1
        self.statistics()
        if parameters['emb_method']:
            self.load_embeds(self.params['word_dim'])
        else:
            assert parameters['dataset'] == 'docred'
            self.load_doc_embeds()

        print(' --> Words + Pre-trained: {:<5}'.format(self.n_words))




