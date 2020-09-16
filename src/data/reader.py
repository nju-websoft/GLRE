#!/usr/bin/env python3
#encoding=utf-8

from collections import OrderedDict
from recordtype import recordtype
import numpy as np


EntityInfo = recordtype('EntityInfo', 'id type mstart mend sentNo')
PairInfo = recordtype('PairInfo', 'type direction cross intrain')


def chunks(l, n, sen_len=None, word_len=None):
    """
    Successive n-sized chunks from l.
    @:param sen_len
    @:param word_len
    """
    res = []
    # print(str(l).encode(encoding='UTF-8', errors='strict'))
    # print(len(l))
    for i in range(0, len(l), n):
        # print(str([l[i:i + n]]).encode(encoding='UTF-8', errors='strict'))
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    if sen_len is not None:
        for i in res:
            a = i[10]
            a_word_len_start = i[8]  # mention start position
            a_word_len_end = i[9]
            b = i[16]
            b_word_len_start = i[14]
            b_word_len_end = i[15]
            for x in a_word_len_start.split(':'):
                assert int(x) <= word_len-1, print(l, '\t', word_len)
            for x in b_word_len_start.split(':'):
                assert int(x) <= word_len-1, print(l, '\t', word_len)
            for x in a_word_len_end.split(':'):
                assert int(x) <= word_len, print(l, '\t', word_len)
            for x in b_word_len_end.split(':'):
                assert int(x) <= word_len, print(l, '\t', word_len)
            for x in a.split(':'):
                assert int(x) <= sen_len-1, print(l, '\t', word_len)
            for x in b.split(':'):
                assert int(x) <= sen_len-1, print(l, '\t', word_len)

            i[8] = ':'.join([str(min(int(x), word_len - 1)) for x in a_word_len_start.split(':')])
            i[14] = ':'.join([str(min(int(x), word_len - 1)) for x in b_word_len_start.split(':')])
            i[9] = ':'.join([str(min(int(x), word_len)) for x in a_word_len_end.split(':')])
            i[15] = ':'.join([str(min(int(x), word_len)) for x in b_word_len_end.split(':')])

            i[10] = ':'.join([str(min(int(x), sen_len - 1)) for x in a.split(':')])
            i[16] = ':'.join([str(min(int(x), sen_len - 1)) for x in b.split(':')])

    return res


def overlap_chunk(chunk=1, lst=None):
    if len(lst) <= chunk:
        return [lst]
    else:
        return [lst[i:i + chunk] for i in range(0, len(lst)-chunk+1, 1)]


def get_distance(e1_sentNo, e2_sentNo):
    distance = 10000
    for e1 in e1_sentNo.split(':'):
        for e2 in e2_sentNo.split(':'):
            distance = min(distance, abs(int(e2) - int(e1)))
    return distance


def read(input_file, documents, entities, relations):
    """
    Read the full document at a time.
    """
    lengths = []
    sents = []
    # relation_have = {}
    entities_cor_id = {}
    with open(input_file, 'r', encoding='utf-8') as infile:
        print("input file ", input_file, "DocRED" in input_file)
        if 'DocRED' in input_file:
            split_n = '||'
        else:
            split_n = '|'
        for line in infile:
            line = line.strip().split('\t')
            pmid = line[0]
            text = line[1]
            # print("pmid:\t" + pmid)
            # print("text:\t", str(text).encode())
            # print("line:\t", str(line).encode())
            entities_dist = []
            sen_len = len(text.split(split_n))
            word_len = sum([len(t.split(' ')) for t in text.split(split_n)])
            if "DocRED" in input_file:
                prs = chunks(line[2:], 18, sen_len, word_len)
            else:  # for CDR
                prs = chunks(line[2:], 17)

            if pmid not in documents:
                documents[pmid] = [t.split(' ') for t in text.split(split_n)]

            if pmid not in entities:
                entities[pmid] = OrderedDict()

            if pmid not in relations:
                relations[pmid] = OrderedDict()

            # max sentence length
            lengths += [max([len(s) for s in documents[pmid]])]
            sents += [len(text.split(split_n))]

            allp = 0
            for p in prs:
                # entities
                if p[5] not in entities[pmid]:
                    entities[pmid][p[5]] = EntityInfo(p[5], p[7], p[8], p[9], p[10])
                    entities_dist.append((p[5], min([int(a) for a in p[8].split(':')])))

                if p[11] not in entities[pmid]:
                    entities[pmid][p[11]] = EntityInfo(p[11], p[13], p[14], p[15], p[16])
                    entities_dist.append((p[11], min([int(a) for a in p[14].split(':')])))

                entity_pair_dis = get_distance(p[10], p[16])
                assert p[0] != 'not_include' or 'DocRED' not in input_file
                if p[0] == 'not_include': # for cdr dataset
                    continue
                if (p[5], p[11]) not in relations[pmid]:
                    relations[pmid][(p[5], p[11])] = [PairInfo(p[0], p[1], entity_pair_dis, p[-1])]
                    assert PairInfo(p[0], p[1], entity_pair_dis, p[-1]) in relations[pmid][(p[5], p[11])]
                    allp += 1
                else:
                    assert PairInfo(p[0], p[1], entity_pair_dis, p[-1]) not in relations[pmid][(p[5], p[11])]
                    relations[pmid][(p[5], p[11])].append(PairInfo(p[0], p[1], entity_pair_dis, p[-1]))
                    # print(p[5]+ "\t" + p[11])
                    # print('duplicates!')

            entities_dist.sort(key=lambda x: x[1], reverse=False)
            entities_cor_id[pmid] = {}
            for coref_id, key in enumerate(entities_dist):
                entities_cor_id[pmid][key[0]] = coref_id + 1
            # assert len(relations[pmid]) == allp

    if "CDR" in input_file:
        todel = []
        for pmid, d in relations.items():
            if not relations[pmid]:
                todel += [pmid]
        print(input_file)
        print("del list", str(todel))
        for pmid in todel:
            del documents[pmid]
            del entities[pmid]
            del relations[pmid]

    return lengths, sents, documents, entities, relations, entities_cor_id
