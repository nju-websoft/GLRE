import torch
from torch import nn
import torch.nn.functional as F

from nnet.modules import EmbedLayer
from utils.tensor_utils import pool


class BaseModel(nn.Module):

    def __init__(self, params, pembeds, loss_weight, sizes, maps, lab2ign):
        super().__init__()

        self.device = torch.device("cuda" if params['gpu'] != -1 else "cpu")
        self.word_embed = EmbedLayer(num_embeddings=sizes['word_size'],
                                     embedding_dim=params['word_dim'],
                                     dropout=0.0,
                                     ignore=None,
                                     freeze=params['freeze_words'],
                                     pretrained=pembeds,
                                     mapping=maps['word2idx'])

        self.ner_emb = EmbedLayer(num_embeddings=sizes['type_size'],
                                  embedding_dim=params['type_dim'], dropout=0.0, ignore=0)


        if params['finaldist']:
            self.dist_embed_dir = EmbedLayer(num_embeddings=20, embedding_dim=params['dist_dim'],
                                             dropout=0.0,
                                             ignore=10,
                                             freeze=False,
                                             pretrained=None,
                                             mapping=None)


        self.loss = nn.BCEWithLogitsLoss(reduction='none')


        # hyper-parameters for tuning
        self.dist_dim = params['dist_dim']
        self.type_dim = params['type_dim']
        self.drop_i = params['drop_i']
        self.drop_o = params['drop_o']
        self.gradc = params['gc']
        self.learn = params['lr']
        self.reg = params['reg']
        self.out_dim = params['out_dim']
        self.batch_size = params['batch']

        # other parameters
        self.mappings = {'word': maps['word2idx'], 'type': maps['type2idx']}
        self.inv_mappings = {'word': maps['idx2word'], 'type': maps['idx2type']}
        self.word_dim = params['word_dim']
        self.lstm_dim = params['lstm_dim']
        self.rel_size = sizes['rel_size']
        self.ignore_label = lab2ign

        self.finaldist = params['finaldist']
        self.dataset = params['dataset']

    def input_layer(self, words_, ner_):
        """
        Word/NER/COREF Embedding Layer
        """
        word_vec = self.word_embed(words_)
        ner_vec = self.ner_emb(ner_)
        # return torch.cat([word_vec, coref_vec, ner_vec], dim=-1)
        return torch.cat([word_vec, ner_vec], dim=-1)

    @staticmethod
    def merge_tokens(info, enc_seq, type="mean"):
        """
        Merge tokens into mentions;
        Find which tokens belong to a mention (based on start-end ids) and average them
        @:param enc_seq all_word_len * dim  4469*192
        """
        mentions = []
        for i in range(info.shape[0]):
            if type == "max":
                mention = torch.max(enc_seq[info[i, 2]: info[i, 3], :], dim=-2)[0]
            else:  # mean
                mention = torch.mean(enc_seq[info[i, 2]: info[i, 3], :], dim=-2)
            mentions.append(mention)
        mentions = torch.stack(mentions)
        return mentions

    @staticmethod
    def merge_mentions(info, mentions, type="mean"):
        """
        Merge mentions into entities;
        Find which rows (mentions) have the same entity id and average them
        """
        m_ids, e_ids = torch.broadcast_tensors(info[:, 0].unsqueeze(0),
                                               torch.arange(0, max(info[:, 0]) + 1).unsqueeze(-1).to(info.device))
        index_f = torch.ne(m_ids, e_ids).bool().to(info.device)
        entities = []
        for i in range(index_f.shape[0]):
            entity = pool(mentions, index_f[i, :].unsqueeze(-1), type=type)
            entities.append(entity)
        entities = torch.stack(entities)
        return entities

    @staticmethod
    def select_pairs(nodes_info, idx, dataset='docred'):
        """
        Select (entity node) pairs for classification based on input parameter restrictions (i.e. their entity type).
        """
        sel = torch.zeros(nodes_info.size(0), nodes_info.size(1), nodes_info.size(1)).to(nodes_info.device)
        a_ = nodes_info[..., 0][:, idx[0]]
        b_ = nodes_info[..., 0][:, idx[1]]
        if dataset == 'cdr':
            c_ = nodes_info[..., 1][:, idx[0]]
            d_ = nodes_info[..., 1][:, idx[1]]
            condition1 = torch.eq(a_, 0) & torch.eq(b_, 0) & torch.ne(idx[0], idx[1])  # needs to be an entity node (id=0)
            condition2 = torch.eq(c_, 1) & torch.eq(d_, 2)  # h=medicine, t=disease
            sel = torch.where(condition1 & condition2, torch.ones_like(sel), sel)
        else:
            condition1 = torch.eq(a_, 0) & torch.eq(b_, 0) & torch.ne(idx[0], idx[1])
            sel = torch.where(condition1, torch.ones_like(sel), sel)
        return sel.nonzero().unbind(dim=1), sel.nonzero()[:, 0]

    def count_predictions(self, y, t):
        """
        Count number of TP, FP, FN, TN for each relation class
        """
        label_num = torch.as_tensor([self.rel_size]).long().to(self.device)
        ignore_label = torch.as_tensor([self.ignore_label]).long().to(self.device)

        mask_t = torch.eq(t, ignore_label).view(-1)          # where the ground truth needs to be ignored
        mask_p = torch.eq(y, ignore_label).view(-1)          # where the predicted needs to be ignored

        true = torch.where(mask_t, label_num, t.view(-1).long().to(self.device))  # ground truth
        pred = torch.where(mask_p, label_num, y.view(-1).long().to(self.device))  # output of NN

        tp_mask = torch.where(torch.eq(pred, true), true, label_num)
        fp_mask = torch.where(torch.ne(pred, true), pred, label_num)
        fn_mask = torch.where(torch.ne(pred, true), true, label_num)

        tp = torch.bincount(tp_mask, minlength=self.rel_size + 1)[:self.rel_size]
        fp = torch.bincount(fp_mask, minlength=self.rel_size + 1)[:self.rel_size]
        fn = torch.bincount(fn_mask, minlength=self.rel_size + 1)[:self.rel_size]
        tn = torch.sum(mask_t & mask_p)
        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'ttotal': t.shape[0]}

    def estimate_loss(self, pred_pairs, truth, multi_truth):
        """
        Softmax cross entropy loss.
        Args:
            pred_pairs (Tensor): Un-normalized pairs (# pairs, classes)
            multi_truth (Tensor) : (#pairs, rel_size)

        Returns: (Tensor) loss, (Tensors) TP/FP/FN
        """
        multi_mask = torch.sum(torch.ne(multi_truth, 0), -1).gt(0)
        # assert (multi_mask == 1).all()
        pred_pairs = pred_pairs[multi_mask]
        multi_truth = multi_truth[multi_mask]
        truth = truth[multi_mask]
        # label smoothing
        # multi_truth -= self.smoothing * ( multi_truth  - 1. / multi_truth.shape[-1])
        loss = torch.sum(self.loss(pred_pairs, multi_truth)) / (
                torch.sum(multi_mask) * self.rel_size)

        predictions = torch.sigmoid(pred_pairs).data.argmax(dim=1)
        stats = self.count_predictions(predictions, truth)
        return loss, stats, predictions, pred_pairs, multi_truth, multi_mask, truth