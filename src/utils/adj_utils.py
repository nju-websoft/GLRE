import numpy as np
import scipy.sparse as sp
import time
import pickle
import torch


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))
    adj_normalized = normalize_adj(adj)
    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_mxs_to_torch_sparse_tensor(sparse_mxs):
    """
    Convert a list of scipy sparse matrix to a torch sparse tensor.
    :param sparse_mxs: [sparse_mx] adj
    :return:
    """
    max_shape = 0
    for mx in sparse_mxs:
        max_shape = max(max_shape, mx.shape[0])
    b_index = []
    row_index = []
    col_index = []
    value = []
    for index, sparse_mx in enumerate(sparse_mxs):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        b_index.extend([index] * len(sparse_mx.row))
        row_index.extend(sparse_mx.row)
        col_index.extend(sparse_mx.col)
        value.extend(sparse_mx.data)
    indices = torch.from_numpy(
        np.vstack((b_index, row_index, col_index)).astype(np.int64))
    values = torch.FloatTensor(value)
    shape = torch.Size([len(sparse_mxs), max_shape, max_shape])
    return torch.sparse.FloatTensor(indices, values, shape)


def convert_3dsparse_to_4dsparse(sparse_mxs):
    """
    :param sparse_mxs: [3d_sparse_tensor]
    :return:
    """
    max_shape = 0
    for mx in sparse_mxs:
        max_shape = max(max_shape, mx.shape[1])
    b_index = []
    indexs = []
    values = []
    for index, sparse_mx in enumerate(sparse_mxs):
        indices_ = sparse_mx._indices()
        values_ = sparse_mx._values()
        b_index.extend([index] * values_.shape[0])
        indexs.append(indices_)
        values.append(values_)
    indexs = torch.cat(indexs, dim=-1)
    b_index = torch.as_tensor(b_index)
    b_index = b_index.unsqueeze(0)
    indices = torch.cat([b_index, indexs], dim=0)
    values = torch.cat(values, dim=-1)
    shape = torch.Size([len(sparse_mxs), sparse_mxs[0].shape[0], max_shape, max_shape])
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == "__main__":
    pass
