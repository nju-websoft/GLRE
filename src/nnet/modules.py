#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class EmbedLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout, ignore=None, freeze=False, pretrained=None, mapping=None):
        """
        Args:
            num_embeddings: (tensor) number of unique items
            embedding_dim: (int) dimensionality of vectors
            dropout: (float) dropout rate
            trainable: (bool) train or not
            pretrained: (dict) pretrained embeddings
            mapping: (dict) mapping of items to unique ids
        """
        super(EmbedLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.freeze = freeze
        self.ignore = ignore

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=ignore)
        self.embedding.weight.requires_grad = not freeze

        if pretrained:
            self.load_pretrained(pretrained, mapping)

        self.drop = nn.Dropout(dropout)

    def load_pretrained(self, pretrained, mapping):
        """
        Args:
            weights: (dict) keys are words, values are vectors
            mapping: (dict) keys are words, values are unique ids
            trainable: (bool)

        Returns: updates the embedding matrix with pre-trained embeddings
        """
        # if self.freeze:
        pret_embeds = torch.zeros((self.num_embeddings, self.embedding_dim))
        # else:
        # pret_embeds = nn.init.normal_(torch.empty((self.num_embeddings, self.embedding_dim)))
        for word in mapping.keys():
            if word in pretrained:
                pret_embeds[mapping[word], :] = torch.from_numpy(pretrained[word])
            elif word.lower() in pretrained:
                pret_embeds[mapping[word], :] = torch.from_numpy(pretrained[word.lower()])
        self.embedding = self.embedding.from_pretrained(pret_embeds, freeze=self.freeze) # , padding_idx=self.ignore

    def forward(self, xs):
        """
        Args:
            xs: (tensor) batchsize x word_ids

        Returns: (tensor) batchsize x word_ids x dimensionality
        """
        embeds = self.embedding(xs)
        if self.drop.p > 0:
            embeds = self.drop(embeds)

        return embeds


class Encoder(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers, bidirectional, dropout):
        """
        Wrapper for LSTM encoder
        Args:
            input_size (int): the size of the input features
            rnn_size (int):
            num_layers (int):
            bidirectional (bool):
            dropout (float):
        Returns: outputs, last_outputs
        - **outputs** of shape `(batch, seq_len, hidden_size)`:
          tensor containing the output features `(h_t)`
          from the last layer of the LSTM, for each t.
        - **last_outputs** of shape `(batch, hidden_size)`:
          tensor containing the last output features
          from the last layer of the LSTM, for each t=seq_len.
        """
        super(Encoder, self).__init__()

        self.enc = nn.LSTM(input_size=input_size,
                           hidden_size=rnn_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)

        # the dropout "layer" for the output of the RNN
        self.drop = nn.Dropout(dropout)

        # define output feature size
        self.feature_size = rnn_size
        self.rnn_size = rnn_size

        if bidirectional:
            self.feature_size *= 2

        self.num_layers = num_layers
        self.bidirectional = bidirectional

    @staticmethod
    def sort(lengths):
        sorted_len, sorted_idx = lengths.sort()  # indices that result in sorted sequence
        _, original_idx = sorted_idx.sort(0, descending=True)
        reverse_idx = torch.linspace(lengths.size(0) - 1, 0, lengths.size(0)).long()  # for big-to-small

        return sorted_idx, original_idx, reverse_idx

    def forward(self, embeds, lengths, hidden=None):
        """
        This is the heart of the model. This function, defines how the data
        passes through the network.
        Args:
            embs (tensor): word embeddings
            lengths (list): the lengths of each sentence
        Returns: the logits for each class
        """
        # sort sequence
        sorted_idx, original_idx, reverse_idx = self.sort(lengths)
        # pad - sort - pack
        embeds = nn.utils.rnn.pad_sequence(embeds, batch_first=True, padding_value=0)
        embeds = embeds[sorted_idx][reverse_idx]  # big-to-small
        embeds = self.drop(embeds)  # apply dropout for input
        packed = pack_padded_sequence(embeds, list(lengths[sorted_idx][reverse_idx].data), batch_first=True)

        self.enc.flatten_parameters()
        out_packed, (h, c) = self.enc(packed, hidden)
        if self.bidirectional:
            h = h.reshape(self.num_layers, 2, -1, self.rnn_size)[-1, :, :, :]
        else:
            h = h.reshape(self.num_layers, 1, -1, self.rnn_size)[-1, :, :, :]

        # unpack
        outputs, _ = pad_packed_sequence(out_packed, batch_first=True)

        # apply dropout to the outputs of the RNN
        outputs = self.drop(outputs)

        # unsort the list
        outputs = outputs[reverse_idx][original_idx][reverse_idx]
        h = h.permute(1, 0, 2)[reverse_idx][original_idx][reverse_idx]
        return outputs, h.view(-1, self.feature_size)


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, num_units, nlayers, bidir, dropout):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        self.init_hidden = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.init_c = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

        self.dropout = LockedDropout(dropout)
        self.nlayers = nlayers

        self.reset_parameters()

    @staticmethod
    def sort(lengths):
        sorted_len, sorted_idx = lengths.sort()  # indices that result in sorted sequence
        _, original_idx = sorted_idx.sort(0, descending=True)
        reverse_idx = torch.linspace(lengths.size(0) - 1, 0, lengths.size(0)).long()  # for big-to-small

        return sorted_idx, original_idx, reverse_idx

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths):
        # sort sequence
        sorted_idx, original_idx, reverse_idx = self.sort(input_lengths)

        # pad - sort - pack
        input = nn.utils.rnn.pad_sequence(input, batch_first=True, padding_value=0)
        input = input[sorted_idx][reverse_idx]  # big-to-small

        bsz, slen = input.size(0), input.size(1)
        output = input
        lens = list(input_lengths[sorted_idx][reverse_idx].data)
        outputs = []
        hiddens = []

        for i in range(self.nlayers):
            hidden, c = self.get_init(bsz, i)

            output = self.dropout(output)

            output = pack_padded_sequence(output, lens, batch_first=True)
            self.rnns[i].flatten_parameters()
            output, (hidden, cn) = self.rnns[i](output, (hidden, c))

            output, _ = pad_packed_sequence(output, batch_first=True)
            if output.size(1) < slen:  # used for parallel
                padding = Variable(output.data.new(1, 1, 1).zero_())
                output = torch.cat([output, padding.expand(output.size(0), slen - output.size(1), output.size(2))], dim=1)

            hiddens.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            outputs.append(output)

        # assert torch.equal(outputs[-1][reverse_idx][original_idx][reverse_idx], inpu)
        return outputs[-1][reverse_idx][original_idx][reverse_idx], hiddens[-1][reverse_idx][original_idx][reverse_idx]


class Classifier(nn.Module):
    def __init__(self, in_size, out_size, dropout):
        """
        Args:
            in_size: input tensor dimensionality
            out_size: outpout tensor dimensionality
            dropout: dropout rate
        """
        super(Classifier, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.lin = nn.Linear(in_features=in_size,
                             out_features=out_size,
                             bias=True)

    def forward(self, xs):
        """
        Args:
            xs: (tensor) batchsize x * x features

        Returns: (tensor) batchsize x * x class_size
        """
        if self.drop.p > 0:
            xs = self.drop(xs)

        xs = self.lin(xs)
        return xs


if __name__ == "__main__":
    lengths = torch.tensor([1,2,3,3,2,1,3,4,2,1,1,5])
    vv = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12])
    sorted_len, sorted_idx = lengths.sort()  # indices that result in sorted sequence
    print("sorted_len", sorted_len)
    print("sorted_idx", sorted_idx)
    _, original_idx = sorted_idx.sort(0, descending=True)
    print("original_idx", original_idx)
    reverse_idx = torch.linspace(lengths.size(0) - 1, 0, lengths.size(0)).long()  # for big-to-small
    print("reverse_idx", reverse_idx)

    vs = vv[sorted_idx][reverse_idx]
    print(vs)

    print(vs[reverse_idx][original_idx][reverse_idx])







