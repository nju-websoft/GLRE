import torch
from torch import nn
from transformers import *
import numpy as np
import os

from transformers import AlbertConfig, AlbertModel, AlbertTokenizer

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),  # bertModel
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "xlm": (XLMConfig, XLMModel, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer)
}

class transformers_word_handle():

    MASK = '[MASK]'
    CLS = "[CLS]"
    SEP = "[SEP]"

    def __init__(self, model_type, model_name, dataset="docred"):
        super().__init__()
        self.model_name = model_name
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if dataset=="docred":
            if self.model_name == 'bert-large-uncased-whole-word-masking' and os.path.exists('./bert_large/'):
                print("find large bert")
                self.tokenizer = tokenizer_class.from_pretrained('./bert_large/')
            elif self.model_name == 'bert-base-uncased' and os.path.exists('./bert_base/'):
                print("find base bert")
                self.tokenizer = tokenizer_class.from_pretrained('./bert_base/')
            else:
                self.tokenizer = tokenizer_class.from_pretrained(self.model_name)
        else: # for cdr dataset
            if self.model_name == 'bert-large-uncased-whole-word-masking' and os.path.exists('./biobert_large/'):
                print("find large biobert")
                self.tokenizer = tokenizer_class.from_pretrained('./biobert_large/')
            elif self.model_name == 'bert-base-uncased' and os.path.exists('./biobert_base/'):
                print("find base biobert")
                self.tokenizer = tokenizer_class.from_pretrained('./biobert_base/')
            elif 'albert' in self.model_name or self.model_name == 'xlnet-large-cased':
                self.tokenizer = tokenizer_class.from_pretrained(self.model_name)
            else:
                print(self.model_name)
                print("can't find biobert")
                exit(1)
        # self.model = model_class.from_pretrained(model_name).to(device)
        self.max_len = 512  # self.model.embeddings.position_embeddings.weight.size(0)
        # self.dim = self.model.embeddings.position_embeddings.weight.size(1)
        if model_type == "xlnet":
            global MASK, CLS, SEP
            MASK = '<mask>'
            CLS = "<cls>"
            SEP = "<sep>"

    def tokenize(self, text, masked_idxs=None):
        tokenized_text = self.tokenizer.tokenize(text)
        if masked_idxs is not None:
            for idx in masked_idxs:
                tokenized_text[idx] = self.MASK
        # prepend [CLS] and append [SEP]
        if self.model_name == 'xlnet-large-cased':  # xlnet
            tokenized = tokenized_text + [self.SEP] + [self.CLS]
        else: # bert
            tokenized = [self.CLS] + tokenized_text + [self.SEP]
        return tokenized

    def tokenize_to_ids(self, text, masked_idxs=None, pad=True):
        tokens = self.tokenize(text, masked_idxs)
        return self.convert_tokens_to_ids(tokens, pad=pad)

    def convert_tokens_to_ids(self, tokens, pad=True):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor([token_ids])
        assert ids.size(1) < self.max_len
        if pad:
            padded_ids = torch.zeros(1, self.max_len).to(ids)
            padded_ids[0, :ids.size(1)] = ids
            mask = torch.zeros(1, self.max_len).to(ids)
            mask[0, :ids.size(1)] = 1
            return padded_ids, mask
        else:
            return ids

    def flatten(self, list_of_lists):
        for list in list_of_lists:
            for item in list:
                yield item

    def subword_tokenize(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subwords, flanked by the special symbols required
                by Bert (CLS and SEP).
            - An array of indices into the list of subwords, indicating
                that the corresponding subword is the start of a new
                token. For example, [1, 3, 4, 7] means that the subwords
                1, 3, 4, 7 are token starts, while all other subwords
                (0, 2, 5, 6, 8...) are in or at the end of tokens.
                This list allows selecting Bert hidden states that
                represent tokens, which is necessary in sequence
                labeling.
        """
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = list(map(len, subwords))
        if self.model_name == 'xlnet-large-cased':  # xlnet 系列
            subwords = list(self.flatten(subwords))[:509] + [self.SEP] + [self.CLS]
        else: # bert 系列
            subwords = [self.CLS] + list(self.flatten(subwords))[:509] + [self.SEP]
        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        token_start_idxs[token_start_idxs > 509] = 509
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries and convert subwords into IDs.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subword IDs, including IDs of the special
                symbols (CLS and SEP) required by Bert.
            - A mask indicating padding tokens.
            - An array of indices into the list of subwords. See
                doc of subword_tokenize.
        """
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords)
        token_starts = np.zeros((1, self.max_len))
        token_starts[0, token_start_idxs] = 1
        return subword_ids.numpy(), mask.numpy(), token_starts

    def segment_ids(self, segment1_len, segment2_len):
        ids = [0] * segment1_len + [1] * segment2_len
        return torch.tensor([ids])