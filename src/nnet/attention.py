#!/usr/bin/env python3
import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math


class Dot_Attention(nn.Module):
    """
    Adaptation from "Attention is all you need".
    Here the query is the target pair and the keys/values are the words of the sentence.
    The dimensionality of the queries and the values should be the same.
    """
    def __init__(self, input_size, device=-1, scale=False):

        super(Dot_Attention, self).__init__()

        self.softmax = nn.Softmax(dim=2)
        self.scale = scale
        if scale:
            self.sc = 1.0 / math.sqrt(input_size)
        self.device = device

    def create_mask(self, alpha, size_, lengths, idx_):
        """ Put 1 in valid tokens """
        mention_sents = torch.index_select(lengths, 0, idx_[:, 4])

        # mask padded words (longer that sentence length)
        tempa = torch.arange(size_).unsqueeze(0).repeat(alpha.shape[0], 1).to(self.device)
        mask_a = torch.ge(tempa, mention_sents[:, None])

        # mask tokens that are used as queries
        tempb = torch.arange(lengths.size(0)).unsqueeze(0).repeat(alpha.shape[0], 1).to(self.device)  # m x sents
        sents = torch.where(torch.lt(tempb, idx_[:, 4].unsqueeze(1)),
                            lengths.unsqueeze(0).repeat(alpha.shape[0], 1),
                            torch.zeros_like(lengths.unsqueeze(0).repeat(alpha.shape[0], 1)))

        total_l = torch.cumsum(sents, dim=1)[:, -1]
        mask_b = torch.ge(tempa, (idx_[:, 2] - total_l)[:, None]) & torch.lt(tempa, (idx_[:, 3] - total_l)[:, None])

        mask = ~(mask_a | mask_b)
        del tempa, tempb, total_l
        return mask

    def forward(self, queries, values, idx, lengths):
        """
        queries: <mention_size, dim> 56, 384
        values: <mention_size, sen_len, dim> 56, 97, 384
        idx:  info (Tensor, 5 columns) entity_id, entity_type, start_wid, end_wid, sentence_id
        lengths:  word_sec (Tensor) number of words per sentence
        a = softmax( q * H^T )
        v = a * H
        """
        alpha = torch.matmul(queries.unsqueeze(1), values.transpose(1, 2))  # men_size * 1 * sen_len
        if self.scale:
            alpha = alpha * self.sc

        mask_ = self.create_mask(alpha, values.size(1), lengths, idx)
        alpha = torch.where(mask_.unsqueeze(1),
                            alpha,
                            torch.as_tensor([float('-inf')]).to(self.device))
        alpha = self.softmax(alpha)
        alpha = torch.squeeze(alpha, 1)
        return alpha


class ScaleDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaleDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        :param q: query [b, l_q, d_q]
        :param k: keys [b, l_k, d_k]
        :param v: values [b, l_v, d_v]， k=v
        :param scale:
        :param attn_mask: masking  [b, l_q, l_k]
        :return:
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -1e12)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        # torch.nn.init.eye_(self.linear_k.weight)
        # torch.nn.init.eye_(self.linear_v.weight)
        # torch.nn.init.eye_(self.linear_q.weight)

        self.dot_product_attention = ScaleDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)
        entity_size = query.size(1)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.reshape(batch_size, -1, num_heads, dim_per_head).transpose(1,2).reshape(batch_size*num_heads, -1, dim_per_head)
        value = value.reshape(batch_size, -1, num_heads, dim_per_head).transpose(1,2).reshape(batch_size*num_heads, -1, dim_per_head)
        query = query.reshape(batch_size, -1, num_heads,  dim_per_head).transpose(1,2).reshape(batch_size*num_heads, -1, dim_per_head)
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
            # scaled dot product attention
        scale = (key.size(-1)) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.reshape(batch_size,num_heads, -1,  dim_per_head).transpose(1,2).reshape(batch_size, -1, num_heads*dim_per_head)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = output.reshape(batch_size, entity_size, entity_size, -1 )
        output = self.layer_norm(output)

        return output, attention

class MultiHeadAttention2(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaleDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)
        entity_size = query.size(1)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.reshape(batch_size, -1, num_heads, dim_per_head).transpose(1,2).reshape(batch_size*num_heads, -1, dim_per_head)
        value = value.reshape(batch_size, -1, num_heads, dim_per_head).transpose(1,2).reshape(batch_size*num_heads, -1, dim_per_head)
        query = query.reshape(batch_size, -1, num_heads,  dim_per_head).transpose(1,2).reshape(batch_size*num_heads, -1, dim_per_head)
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
            # scaled dot product attention
        scale = (key.size(-1)) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.reshape(batch_size,num_heads, -1,  dim_per_head).transpose(1,2).reshape(batch_size, -1, num_heads*dim_per_head)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = output.reshape(batch_size, entity_size, entity_size, -1 )
        output = self.layer_norm(output)

        return output, attention


class SelfAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        # self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)), requires_grad=True)

    def forward(self, input, memory, mask):
        # input = self.dropout(input)
        # memory = self.dropout(memory)
        batch_size = input.shape[0]
        enttiy_size = input.shape[1]
        input = input.reshape(batch_size, enttiy_size*enttiy_size, -1)
        memory = memory.reshape(batch_size, enttiy_size*enttiy_size, -1)
        mask = mask.reshape(batch_size,-1)
        input_dot = self.input_linear(input)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + cross_dot
        att = att - 1e30 * (~ mask[:, None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)

        return torch.cat([input, output_one], dim=-1).reshape(batch_size, enttiy_size, enttiy_size, -1)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SimpleEncoder(nn.Module):
    """
    takes (batch_size, seq_len, embed_dim) as inputs
    calculate MASK, POSITION_ENCODING
    """
    def __init__(self, embed_dim, head=4, layer=1, dropout=0.1):
        super(SimpleEncoder, self).__init__()
        d_ff = 2 * embed_dim

        self.position = PositionalEncoding(embed_dim, dropout)
        attn = MultiHeadedAttention(head, embed_dim)
        ff = PositionwiseFeedForward(embed_dim, d_ff)
        self.encoder = Encoder(EncoderLayer(embed_dim, attn, ff, dropout), layer)

    def forward(self, x, mask):
        mask = mask.unsqueeze(-2)
        x = self.position(x)
        x = self.encoder(x, mask)
        return x


if __name__ == '__main__':
    encoder = SimpleEncoder(300, 4, 2)
    inputs = torch.zeros(1000,50,300)
    mask = torch.ones(1000, 50)
    mask[:10,30:] = 0
    mask[20:30,20:] = 0
    print(mask)
    lens  = [10] * 1000
    out = encoder(inputs, mask)
    print(out.size())
    print(out[0])
    print(out[-1])