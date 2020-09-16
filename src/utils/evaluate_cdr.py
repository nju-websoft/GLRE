# -*- coding: utf-8 -*-
"""
python evaluate_cdr.py --gold ../data/CDR/processed/test.gold --pred ../results/cdr-dev/EOG/test.preds --label 1:CID:2
"""

import argparse


def prf(tp, fp, fn):
    micro_p = float(tp) / (tp + fp) if (tp + fp != 0) else 0.0
    micro_r = float(tp) / (tp + fn) if (tp + fn != 0) else 0.0
    micro_f = ((2 * micro_p * micro_r) / (micro_p + micro_r)) if micro_p != 0.0 and micro_r != 0.0 else 0.0
    return [micro_p, micro_r, micro_f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', type=str)
    parser.add_argument('--pred', type=str)
    parser.add_argument('--label', type=str)
    args = parser.parse_args()

    with open(args.pred) as pred, open(args.gold) as gold:
        preds_all = set()
        preds_intra = set()
        preds_inter = set()
        preds_inter_1_3 = set()
        preds_inter_3_max = set()

        golds_all = set()
        golds_intra = set()
        golds_inter = set()
        golds_inter_1_3 = set()
        golds_inter_3_max = set()

        for line in pred:
            line = line.rstrip().split('|')
            if line[5] == args.label:

                if (line[0], line[1], line[2], line[5]) not in preds_all:
                    preds_all.add((line[0], line[1], line[2], line[5]))

                if ((line[0], line[1], line[2], line[5]) not in preds_inter) and (line[3] == 'CROSS' or line[3] != '0'):
                    preds_inter.add((line[0], line[1], line[2], line[5]))

                if ((line[0], line[1], line[2], line[5]) not in preds_inter_1_3) and (line[3] == '1' or line[3] == '2'):
                    preds_inter_1_3.add((line[0], line[1], line[2], line[5]))

                if ((line[0], line[1], line[2], line[5]) not in preds_inter_3_max) and (line[3] != '0' and line[3] != '1' and line[3] != '2'):
                    preds_inter_3_max.add((line[0], line[1], line[2], line[5]))

                if ((line[0], line[1], line[2], line[5]) not in preds_intra) and (line[3] == 'NON-CROSS' or line[3] == '0'):
                    preds_intra.add((line[0], line[1], line[2], line[5]))

        for line2 in gold:
            line2 = line2.rstrip().split('|')

            if line2[4] == args.label:

                if (line2[0], line2[1], line2[2], line2[4]) not in golds_all:
                    golds_all.add((line2[0], line2[1], line2[2], line2[4]))

                if ((line2[0], line2[1], line2[2], line2[4]) not in golds_inter) and (line2[3] == 'CROSS' or line2[3] != '0'):
                    golds_inter.add((line2[0], line2[1], line2[2], line2[4]))

                if ((line2[0], line2[1], line2[2], line2[4]) not in golds_inter_1_3) and (line2[3] == '1' or line2[3] == '2'):
                    golds_inter_1_3.add((line2[0], line2[1], line2[2], line2[4]))

                if ((line2[0], line2[1], line2[2], line2[4]) not in golds_inter_3_max) and (
                        line2[3] != '0' and line2[3] != '1' and line2[3] != '2'):
                    golds_inter_3_max.add((line2[0], line2[1], line2[2], line2[4]))

                if ((line2[0], line2[1], line2[2], line2[4]) not in golds_intra) and (line2[3] == 'NON-CROSS' or line2[3] == '0'):
                    golds_intra.add((line2[0], line2[1], line2[2], line2[4]))

        tp = len([a for a in preds_all if a in golds_all])
        tp_intra = len([a for a in preds_intra if a in golds_intra])
        tp_inter = len([a for a in preds_inter if a in golds_inter])
        tp_inter_1_3 = len([a for a in preds_inter_1_3 if a in golds_inter_1_3])
        tp_inter_3_max = len([a for a in preds_inter_3_max if a in golds_inter_3_max])

        fp = len([a for a in preds_all if a not in golds_all])
        fp_intra = len([a for a in preds_intra if a not in golds_intra])
        fp_inter = len([a for a in preds_inter if a not in golds_inter])
        fp_inter_1_3 = len([a for a in preds_inter_1_3 if a not in golds_inter_1_3])
        fp_inter_3_max = len([a for a in preds_inter_3_max if a not in golds_inter_3_max])

        fn = len([a for a in golds_all if a not in preds_all])
        fn_intra = len([a for a in golds_intra if a not in preds_intra])
        fn_inter = len([a for a in golds_inter if a not in preds_inter])
        fn_inter_1_3 = len([a for a in golds_inter_1_3 if a not in preds_inter_1_3])
        fn_inter_3_max = len([a for a in golds_inter_3_max if a not in preds_inter_3_max])

        r1 = prf(tp, fp, fn)
        r2 = prf(tp_intra, fp_intra, fn_intra)
        r3 = prf(tp_inter, fp_inter, fn_inter)
        r4 = prf(tp_inter_1_3, fp_inter_1_3, fn_inter_1_3)
        r5 = prf(tp_inter_3_max, fp_inter_3_max, fn_inter_3_max)

        print('                                          TOTAL\tTP\tFP\tFN')
        print('Overall P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r1[0], r1[1], r1[2],
                                                                                               tp + fn, tp, fp, fn))
        print('INTRA P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r2[0], r2[1], r2[2],
                                                                                             tp_intra + fn_intra,
                                                                                             tp_intra, fp_intra,
                                                                                             fn_intra))
        print('INTER P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r3[0], r3[1], r3[2],
                                                                                             tp_inter + fn_inter,
                                                                                             tp_inter, fp_inter,
                                                                                             fn_inter))
        print('INTER_[1,3) P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r4[0], r4[1], r4[2],
                                                                              tp_inter_1_3 + fn_inter_1_3,
                                                                              tp_inter_1_3, fp_inter_1_3,
                                                                              fn_inter_1_3))
        print('INTER_[3,max) P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r5[0], r5[1], r5[2],
                                                                              tp_inter_3_max + fn_inter_3_max,
                                                                              tp_inter_3_max, fp_inter_3_max,
                                                                              fn_inter_3_max))

if __name__ == "__main__":
    main()
