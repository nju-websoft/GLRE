"""
python docRedProcess.py --input_file ../data/DocRED/train_annotated.json \
                       --output_file ../data/DocRED/processed/train_annotated.data \
"""
import json

max_length = 512
max_sen_length = 200
max_sen_cnt = 36
char2id = json.load(open("../data/DocRED/char2id.json", encoding="utf-8"))

fact_in_dev_train = set([])


def main(input_file, output_file, suffix):
    ori_data = json.load(open(input_file))
    doc_id = -1
    data_out = open(output_file, 'w', encoding="utf-8")

    for i in range(len(ori_data)):
        doc_id += 1
        print("docid", doc_id)
        towrite_meta = str(doc_id) + "\t"  # pmid
        Ls = [0]
        L = 0
        for x in ori_data[i]['sents']:
            L += len(x)
            Ls.append(L)
        for x_index, x in enumerate(ori_data[i]['sents']):
            for ix_index, ix in enumerate(x):
                if " " in ix:
                    assert ix == " " or ix == "  ", print(ix)
                    ori_data[i]['sents'][x_index][ix_index] = "_"
        towrite_meta += "||".join([" ".join(x) for x in ori_data[i]['sents']])  # txt
        p = " ".join([" ".join(x) for x in ori_data[i]['sents']])

        document_list = []
        for x in ori_data[i]['sents']:
            document_list.append(" ".join(x))

        document = "\n".join(document_list)
        print("gg", str(document))
        assert "   " not in document
        assert "||" not in p and "\t" not in p

        vertexSet = ori_data[i]['vertexSet']

        for j in range(len(vertexSet)):
            for k in range(len(vertexSet[j])):
                vertexSet[j][k]['name'] = str(vertexSet[j][k]['name']).replace('4.\nStranmillis Road',
                                                                               'Stranmillis Road')
                vertexSet[j][k]['name'] = str(vertexSet[j][k]['name']).replace("\n", "")
        # point position added with sent start position
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet[j])):
                vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

                sent_id = vertexSet[j][k]['sent_id']
                assert sent_id < len(Ls)-1
                sent_id = min(len(Ls)-1, sent_id)
                dl = Ls[sent_id]
                pos1 = vertexSet[j][k]['pos'][0]
                pos2 = vertexSet[j][k]['pos'][1]
                vertexSet[j][k]['pos'] = (pos1 + dl, pos2 + dl)
                vertexSet[j][k]['s_pos'] = (pos1, pos2)

        labels = ori_data[i].get('labels', [])
        train_triple = set([])
        towrite = ""
        for label in labels:
            train_triple.add((label['h'], label['t']))
        na_triple = []
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet)):
                if (j != k):
                    if (j, k) not in train_triple:
                        na_triple.append((j, k))
                        labels.append({'h': j, 'r': 'NA', 't': k})

        for label in labels:
            rel = label['r']  # 'type'
            dir = "L2R"  # no use 'dir'
            head = vertexSet[label['h']]
            tail = vertexSet[label['t']]
            # train_triple.add((label['h'], label['t']))
            cross = find_cross(head, tail)
            towrite = towrite + "\t" + str(rel) + "\t" + str(dir) + "\t" + str(cross) + "\t" + str(
                head[0]['pos'][0]) + "-" + str(head[0]['pos'][1]) + "\t" + str(tail[0]['pos'][0]) + "-" + str(
                tail[0]['pos'][1])

            if suffix == '_train':
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_dev_train.add((n1['name'], n2['name'], rel))  # annotated data
            # gtype = head[0]['type']
            # for g in head:
            #     assert gtype == g['type']

            towrite += "\t" + str(label['h']) + "\t" + '||'.join([g['name'] for g in head]) + "\t" + ":".join([str(g['type']) for g in head]) \
                       + "\t" + ":".join([str(g['pos'][0]) for g in head]) + "\t" + ":".join(
                [str(g['pos'][1]) for g in head]) + "\t" \
                       + ":".join([str(g['sent_id']) for g in head])

            # gtype = tail[0]['type']
            # for g in tail:
            #     assert gtype == g['type']

            towrite += "\t" + str(label['t']) + "\t" + '||'.join([g['name'] for g in tail]) + "\t" + ":".join([str(g['type']) for g in tail]) \
                       + "\t" + ":".join([str(g['pos'][0]) for g in tail]) + "\t" + ":".join(
                [str(g['pos'][1]) for g in tail]) + "\t" \
                       + ":".join([str(g['sent_id']) for g in tail])

            indev_train = False

            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    if suffix == '_dev' or suffix == '_test':
                        if (n1['name'], n2['name'], rel) in fact_in_dev_train:
                            indev_train = True

            towrite += "\t" + str(indev_train)

        towrite += "\n"
        data_out.write(towrite_meta + towrite)
    data_out.close()


def find_cross(head, tail):
    non_cross = False
    for m1 in head:
        for m2 in tail:
            if m1['sent_id'] == m2['sent_id']:
                non_cross = True
    if non_cross:
        return 'NON-CROSS'
    else:
        return 'CROSS'


if __name__ == '__main__':
    main('../data/DocRED/train_annotated.json', '../data/DocRED/processed/train_annotated.data', suffix='_train')
    main('../data/DocRED/dev.json', '../data/DocRED/processed/dev.data', suffix='_dev')
    main('../data/DocRED/test.json', '../data/DocRED/processed/test.data', suffix='_test')
