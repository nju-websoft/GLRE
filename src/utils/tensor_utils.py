import torch
from torch.nn.utils.rnn import pad_sequence


def split_n_pad(nodes, section, pad=0, return_mask=False):
    """
    split tensor and pad
    :param nodes:
    :param section:
    :param pad:
    :return:
    """
    assert nodes.shape[0] == sum(section.tolist()), print(nodes.shape[0], sum(section.tolist()))
    nodes = torch.split(nodes, section.tolist())
    nodes = pad_sequence(nodes, batch_first=True, padding_value=pad)
    if not return_mask:
        return nodes
    else:
        max_v = max(section.tolist())
        temp_ = torch.arange(max_v).unsqueeze(0).repeat(nodes.size(0), 1).to(nodes)
        mask = (temp_ < section.unsqueeze(1))

        # mask = torch.zeros(nodes.size(0), max_v).to(nodes)
        # for index, sec in enumerate(section.tolist()):
        #    mask[index, :sec] = 1
        # assert (mask1==mask).all(), print(mask1)
        return nodes, mask


def rm_pad(input, lens, max_v=None):
    """
    :param input: batch_size * len * dim
    :param lens: batch_size
    :return:
    """
    if max_v is None:
        max_v = lens.max()
    temp_ = torch.arange(max_v).unsqueeze(0).repeat(lens.size(0), 1).to(input.device)
    remove_pad = (temp_ < lens.unsqueeze(1))
    return input[remove_pad]

def rm_pad_between(input, s_lens, e_lens, max_v):
    """
    :param input: batch_size * len * dim
    :param lens: batch_size
    :return:
    """
    temp_ = torch.arange(max_v).unsqueeze(0).repeat(s_lens.size(0), 1).to(input.device)
    # print(temp_ < e_lens.unsqueeze(1))
    remove_pad = (temp_ < e_lens.unsqueeze(1)) & (temp_ >= s_lens.unsqueeze(1))
    return input[remove_pad]


def pool(h, mask, type='max'):
    """

    :param h:  batch_size * max_len * gcn_hidden_size
    :param mask:
    :param type:
    :return:
    """
    if type == 'max':
        h = h.masked_fill(mask, -1e12)
        return torch.max(h, -2)[0]
    elif type == 'avg' or type == "mean":
        h = h.masked_fill(mask, 0)
        return h.sum(-2) / (mask.size(-2) - mask.float().sum(-2))
    elif type == "logsumexp":
        h = h.masked_fill(mask, -1e12)
        return torch.logsumexp(h,-2)
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(-2)


if __name__ == '__main__':
    import numpy as np
    input = torch.randn((2,10,2))
    print(input)
    section = torch.from_numpy(np.asarray([[2,2,6],[2,3,2]]))
    print(section)
    max_v = section[:, 0:3].sum(dim=1).max()
    print(max_v)
    result = rm_pad_between(input, section[:, 0], section[:, 0:2].sum(dim=1), max_v)
    print(result)
    print(input[0, 2:4, :])