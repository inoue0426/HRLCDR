import itertools as it
import os
import time

import numpy as np
import pandas as pd
import pubchempy as pcp
import scipy.sparse as sp
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, mean_squared_error,
                             roc_auc_score)


def update_Adjacency_matrix(A, test_samples):
    m = test_samples.shape[0]
    A_tep = A.copy()
    for i in range(m):
        if test_samples[i, 2] == 1:
            A_tep[int(test_samples[i, 0]), int(test_samples[i, 1])] = 0
    return A_tep


def k_near_graph(sim: torch.Tensor, k: int):
    """
    Calculate the k near graph as feature space adjacency.
    :param sim: similarity matrix, torch.Tensor
    :param k: k, int
    :return: weighted adjacency matrix
    """
    threshold = torch.min(torch.topk(sim, k=k, dim=1).values, dim=1).values.view(
        [-1, 1]
    )
    sim = torch.where(sim.ge(threshold), 1, torch.zeros_like(sim))
    sim = sim - torch.eye(sim.shape[0]).to("cuda:0")
    return sim


def hyper_graph(sima, guanlian, simb):
    a1 = torch.mm(sima, sima)
    A1 = torch.mul(a1, sima)
    A2 = torch.mul(torch.mm(guanlian, guanlian.T), sima)
    A3 = (
        torch.mul(torch.mm(torch.mm(guanlian, simb), guanlian.T), sima)
        + torch.mul(torch.mm(torch.mm(guanlian, simb), guanlian.T), sima).T
    )
    A5 = torch.mm(torch.mm(sima, guanlian), torch.mm(sima, guanlian).T)

    return A1, A2, A3, A1 + A2 + A3, A5, A1 + A2 + A3 + A5


def laplacian_matrix(adjacency_matrix):
    degree_matrix = np.sum(adjacency_matrix, axis=1) + 1
    degree_matrix = np.power(degree_matrix, -0.5)
    diag = np.diag(degree_matrix)
    lp_1 = np.matmul(diag, adjacency_matrix)
    lp = np.matmul(lp_1, diag)
    return lp


def creat_G_and_X_and_models_prarams(kegg, PPI, cell_exp):
    G = []
    X = []
    models_params = []
    pathway_contain = pd.DataFrame(
        np.zeros((34, 2369)), index=kegg.keys(), columns=PPI.columns
    )
    for key, _ in kegg.items():
        demo1 = kegg[key]
        g1 = np.array(PPI.loc[demo1, demo1])

        la = torch.tensor(laplacian_matrix(g1), dtype=torch.float32).to("cuda:0")
        G.append(la)
        cell = np.array(cell_exp.loc[:, demo1]).T
        X.append(torch.tensor(cell, dtype=torch.float32).to("cuda:0"))
        models_params.append(
            {
                "input_size": cell.shape[0],
                "hidden_size": cell.shape[0],
                "output_size": 30,
            }
        )
        pathway_contain.loc[key, demo1] = 1
    return G, X, models_params, pathway_contain


def pcc_score(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: train data, torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: pcc score
    """
    pcc = pearsonr(
        true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy()
    )[0]
    return pcc


def init_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def distribute_compute(
    lr_list,
    wd_list,
    scale_list,
    layer_size,
    sigma_list,
    beta_list,
    workers: int,
    id: int,
):
    all_list = []
    for lr, wd, sc, la, sg, bt in it.product(
        lr_list, wd_list, scale_list, layer_size, sigma_list, beta_list
    ):
        all_list.append([lr, wd, sc, la, sg, bt])
    list = np.array_split(all_list, workers)
    return list[id]


def get_fingerprint(x):
    """
    :param x: drug cid
    :return: fingerprint
    """
    drug_data = pcp.Compound.from_cid(x)
    fingerprint = ""
    for x in drug_data.fingerprint:
        fingerprint += "{:04b}".format(int(x, 16))
    finger = np.array([int(x) for x in fingerprint])
    return finger


def save_fingerprint(cid_list, last_cid, fpath):
    """
    :param cid_list: cid list
    :param last_cid: last cid, -1 begining
    :param fpath: save path
    :return: file save to path
    """
    if last_cid > 0:
        index = np.where(np.array(cid_list) == last_cid)[0].tolist()[0] + 1
    else:
        index = last_cid + 1
    length = len(cid_list)
    if index < length:
        for i in range(index, length):
            cid = cid_list[i]
            fing = get_fingerprint(cid)
            path = fpath + str(cid)
            np.save(path, fing)
            print(cid, "OK.")
            time.sleep(1)
    else:
        print("All compound has finished!")
    return None


def read_fingerprint_cid(path: str):
    """
    :param path: file path
    :return: fingerprint and cid
    """
    file_list = os.listdir(path)
    fingerprint = np.array([], dtype=np.int)
    cid = []
    for name in file_list:
        if name.endswith("npy"):
            fpath = path + name
            fing = np.load(fpath)
            fingerprint = np.hstack((fingerprint, fing))
            cid.append(int(name.split(".")[0]))
    fingerprint = fingerprint.reshape((-1, 920))
    return fingerprint, cid


def common_data_index(data_for_index: np.ndarray, data_for_cmp: np.ndarray):
    """
    :param data_for_index: data for index, numpy array
    :param data_for_cmp: data for compare, numpy array
    :return: index of common data in data for index
    """
    index = np.array([np.where(x in data_for_cmp, 1, 0) for x in data_for_index])
    index = np.where(index == 1)[0]
    return index


def to_coo_matrix(adj_mat):
    """
    :param adj_mat: adj matrix, numpy array
    :return: sparse matrix, sp.coo.coo_matrix
    """
    if not sp.isspmatrix_coo(adj_mat):
        adj_mat = sp.coo_matrix(adj_mat)
    return adj_mat


def mse_loss(true_data: torch.Tensor, predict_data: torch.Tensor, masked: torch.Tensor):
    """
    :param true_data: true data
    :param predict_data: predict data
    :param masked: data mask
    :return: mean square error
    """
    masked = masked.to(torch.bool)
    true_data = torch.masked_select(true_data, masked)
    predict_data = torch.masked_select(predict_data, masked)
    loss_fun = nn.MSELoss()
    loss = loss_fun(predict_data, true_data)
    return loss


def huber_loss(
    true_data: torch.Tensor, predict_data: torch.Tensor, masked: torch.Tensor, hu
):
    """
    :param true_data: true data
    :param predict_data: predict data
    :param masked: data mask
    :return: mean square error
    """
    masked = masked.to(torch.bool)
    true_data = torch.masked_select(true_data, masked)
    predict_data = torch.masked_select(predict_data, masked)
    loss_fun = nn.HuberLoss(delta=hu)
    loss = loss_fun(predict_data, true_data)
    return loss


def giploss(target, prediction, mirna_lap, disease_lap, alpha1, alpha2, masked):
    masked = masked.to(torch.bool)
    true_data = torch.masked_select(target, masked)
    predict_data = torch.masked_select(prediction, masked)
    loss_ls = torch.norm((true_data - predict_data), p="fro") ** 2

    drug_reg = torch.trace(torch.mm(torch.mm(alpha1.T, mirna_lap), alpha1))
    mic_reg = torch.trace(torch.mm(torch.mm(alpha2.T, disease_lap), alpha2))
    graph_reg = 2 ** (-8) * drug_reg + 2 ** (-8) * mic_reg

    loss_sum = loss_ls + graph_reg
    return loss_sum.sum()


def tri_loss(true_data: torch.Tensor, predict_data: torch.Tensor, masked: torch.Tensor):
    masked = masked.to(torch.bool)
    true_data = torch.masked_select(true_data, masked)
    predict_data = torch.masked_select(predict_data, masked)
    loss_fun = nn.BCELoss()
    loss = loss_fun(predict_data, true_data)
    return loss


def ji_loss(
    true_data: torch.Tensor,
    predict_data: torch.Tensor,
    masked: torch.Tensor,
    cell_emb,
    drug_emb,
    cell_sim,
    drug_sim,
):
    masked = masked.to(torch.bool)
    true_data = torch.masked_select(true_data, masked)
    predict_data = torch.masked_select(predict_data, masked)
    loss_fun = nn.BCELoss()
    loss_bce = loss_fun(predict_data, true_data)
    drug_reg = torch.trace(torch.mm(torch.mm(drug_emb.T, drug_sim), drug_emb))
    cell_reg = torch.trace(torch.mm(torch.mm(cell_emb.T, cell_sim), cell_emb))
    loss_tri = loss_bce + 2 ** (-5) * drug_reg + 2 ** (-5) * cell_reg
    return loss_tri


def cross_entropy_loss(
    true_data: torch.Tensor, predict_data: torch.Tensor, masked: torch.Tensor
):
    """
    :param true_data: true data
    :param predict_data: predict data
    :param masked: data mask
    :return: cross entropy loss
    """
    masked = masked.to(torch.bool)
    true_data = torch.masked_select(true_data, masked)
    predict_data = torch.masked_select(predict_data, masked)
    loss_fun = nn.BCELoss()
    # loss = F.binary_cross_entropy_with_logits(predict_data, true_data)
    loss = loss_fun(predict_data, true_data)
    return loss


def re_entropy_loss(
    predict_data: torch.Tensor, true_data: torch.Tensor, masked: torch.Tensor
):
    """
    :param true_data: true data
    :param predict_data: predict data
    :param masked: data mask
    :return: cross entropy loss
    """
    masked = masked.to(torch.bool)
    true_data = torch.masked_select(true_data, masked)
    predict_data = torch.masked_select(predict_data, masked)
    loss_fun = nn.BCELoss()
    # loss = F.binary_cross_entropy_with_logits(predict_data, true_data)
    loss = loss_fun(predict_data, true_data)
    return loss


def mask(positive: sp.coo.coo_matrix, negative: sp.coo.coo_matrix, dtype=int):
    """
    :param positive: positive data
    :param negative: negative data
    :param dtype: return data type
    :return: data mask
    """
    row = np.hstack((positive.row, negative.row))
    col = np.hstack((positive.col, negative.col))
    data = [1] * row.shape[0]
    masked = (
        sp.coo_matrix((data, (row, col)), shape=positive.shape).toarray().astype(dtype)
    )
    masked = torch.from_numpy(masked)
    return masked


def to_tensor(positive, identity=False):
    """
    :param positive: positive sample
    :param identity: if add identity
    :return: tensor
    """
    if identity:
        data = positive + sp.identity(positive.shape[0])
    else:
        data = positive
    data = torch.from_numpy(data.toarray()).float()
    return data


def evaluate_all(true_data: torch.Tensor, predict_data: torch.Tensor):
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    auc = roc_auc(true_data, predict_data)
    ap = ap_score(true_data, predict_data)
    f1, thresholds = f1_score_binary(true_data, predict_data)
    acc = accuracy_binary(true_data, predict_data, thresholds)
    precision = precision_binary(true_data, predict_data, thresholds)
    recall = recall_binary(true_data, predict_data, thresholds)
    mcc = mcc_binary(true_data, predict_data, thresholds)
    return auc, ap, acc, f1, mcc


def evaluate_auc(true_data: torch.Tensor, predict_data: torch.Tensor):
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    auc = roc_auc(true_data, predict_data)
    ap = ap_score(true_data, predict_data)
    return auc, ap


def pcc_score(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: train data, torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: PCC score
    """
    pcc = pearsonr(
        true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy()
    )[0]
    return pcc


def sc_score(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: train data, torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: spearmanr score
    """
    sc = spearmanr(
        true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy()
    )[0]
    return sc


def rmse_score(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: train data, torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: RMSE score
    """
    rmse = np.sqrt(
        mean_squared_error(
            true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy()
        )
    )
    return rmse


def evaluate_regression(true_data: torch.Tensor, predict_data: torch.Tensor):
    pcc = pcc_score(true_data, predict_data)
    sc = sc_score(true_data, predict_data)
    rmse = rmse_score(true_data, predict_data)
    return pcc, sc, rmse


def ap_score(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    area under the precision-recall curve
    :param true_data: train data, torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: ap
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    area = average_precision_score(
        y_true=true_data.detach().cpu().numpy(),
        y_score=predict_data.detach().cpu().numpy(),
    )
    return area


def roc_auc(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: train data, torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: AUC score
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    aucs = roc_auc_score(
        true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy()
    )
    return aucs


def f1_score_binary(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: true data,torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: max F1 score and threshold
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    with torch.no_grad():
        thresholds = torch.unique(predict_data)
    size = torch.tensor(
        [thresholds.size()[0], true_data.size()[0]],
        dtype=torch.int32,
        device=true_data.device,
    )
    ones = torch.ones(
        [size[0].item(), size[1].item()], dtype=torch.float32, device=true_data.device
    )
    zeros = torch.zeros(
        [size[0].item(), size[1].item()], dtype=torch.float32, device=true_data.device
    )
    predict_value = torch.where(
        predict_data.view([1, -1]).ge(thresholds.view([-1, 1])), ones, zeros
    )
    tpn = torch.sum(
        torch.where(predict_value.eq(true_data.view([1, -1])), ones, zeros), dim=1
    )
    tp = torch.sum(torch.mul(predict_value, true_data.view([1, -1])), dim=1)
    two = torch.tensor(2, dtype=torch.float32, device=true_data.device)
    n = torch.tensor(size[1].item(), dtype=torch.float32, device=true_data.device)
    scores = torch.div(
        torch.mul(two, tp), torch.add(n, torch.sub(torch.mul(two, tp), tpn))
    )
    max_f1_score = torch.max(scores)
    threshold = thresholds[torch.argmax(scores)]
    return max_f1_score, threshold


def accuracy_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: acc
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    n = true_data.size()[0]
    ones = torch.ones(n, dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(n, dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tpn = torch.sum(torch.where(predict_value.eq(true_data), ones, zeros))
    score = torch.div(tpn, n)
    return score


def precision_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(
        true_data.size()[0], dtype=torch.float32, device=true_data.device
    )
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tp = torch.sum(torch.mul(true_data, predict_value))
    true_neg = torch.sub(ones, true_data)
    tf = torch.sum(torch.mul(true_neg, predict_value))
    score = torch.div(tp, torch.add(tp, tf))
    return score


def recall_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(
        true_data.size()[0], dtype=torch.float32, device=true_data.device
    )
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tp = torch.sum(torch.mul(true_data, predict_value))
    predict_neg = torch.sub(ones, predict_value)
    fn = torch.sum(torch.mul(predict_neg, true_data))
    score = torch.div(tp, torch.add(tp, fn))
    return score


def mcc_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(
        true_data.size()[0], dtype=torch.float32, device=true_data.device
    )
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    predict_neg = torch.sub(ones, predict_value)
    true_neg = torch.sub(ones, true_data)
    tp = torch.sum(torch.mul(true_data, predict_value))
    tn = torch.sum(torch.mul(true_neg, predict_neg))
    fp = torch.sum(torch.mul(true_neg, predict_value))
    fn = torch.sum(torch.mul(true_data, predict_neg))
    delta = torch.tensor(0.00001, dtype=torch.float32, device=true_data.device)
    score = torch.div(
        (tp * tn - fp * fn),
        torch.add(delta, torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
    )
    return score


def torch_corr(tensor: torch.Tensor, dim=0):
    """
    :param tensor: an 2D torch tensor
    :param dim:
        0 : Calculate row correlation
        1 : Calculate col correlation
    :return: correlation coefficient
    """
    mean = torch.mean(tensor, dim=1 - dim)
    if dim:
        tensor_mean = torch.sub(tensor, mean)
        tensor_cov = torch.mm(torch.t(tensor_mean), tensor_mean)
    else:
        mean = mean.view([mean.size()[0], -1])
        tensor_mean = torch.sub(tensor, mean)
        tensor_cov = torch.mm(tensor_mean, torch.t(tensor_mean))
    diag = torch.diag(tensor_cov)
    diag = torch.sqrt(diag)
    diag = torch.mm(diag.view([-1, 1]), diag.view([1, -1]))
    tensor_corr = torch.div(tensor_cov, diag)
    return tensor_corr


def torch_corr_x_y(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    :param tensor1: a matrix, torch Tensor
    :param tensor2: a matrix, torch Tensor
    :return: corr(tensor1, tensor2)
    """
    assert tensor1.size()[1] == tensor2.size()[1], "Different size!"
    tensor2 = torch.t(tensor2)
    mean1 = torch.mean(tensor1, dim=1).view([-1, 1])
    mean2 = torch.mean(tensor2, dim=0).view([1, -1])
    lxy = torch.mm(torch.sub(tensor1, mean1), torch.sub(tensor2, mean2))
    lxx = torch.diag(
        torch.mm(torch.sub(tensor1, mean1), torch.t(torch.sub(tensor1, mean1)))
    )
    lyy = torch.diag(
        torch.mm(torch.t(torch.sub(tensor2, mean2)), torch.sub(tensor2, mean2))
    )
    std_x_y = torch.mm(torch.sqrt(lxx).view([-1, 1]), torch.sqrt(lyy).view([1, -1]))
    corr_x_y = torch.div(lxy, std_x_y)
    return corr_x_y


def torch_euclidean_dist(tensor: torch.Tensor, dim=0):
    """
    :param tensor: a 2D torch tensor
    :param dim:
        0 : represent row
        1 : represent col
    :return: return euclidean distance
    """
    if dim:
        tensor_mul_tensor = torch.mm(torch.t(tensor), tensor)
    else:
        tensor_mul_tensor = torch.mm(tensor, torch.t(tensor))
    diag = torch.diag(tensor_mul_tensor)
    n_diag = diag.size()[0]
    tensor_diag = diag.repeat([n_diag, 1])
    diag = diag.view([n_diag, -1])
    dist = torch.sub(torch.add(tensor_diag, diag), torch.mul(tensor_mul_tensor, 2))
    dist = torch.sqrt(dist)
    return dist


def torch_dist(tensor: torch.Tensor, p=0 or int):
    """
    :param tensor: an 2D tensor
    :param p: pow
    :return: distance between rows
    """
    size = tensor.size()
    tensor_flatten = torch.flatten(tensor)
    tensor_mat = tensor.repeat([1, 1, size[0]])
    tensor_flatten = tensor_flatten.repeat([1, size[0], 1])
    tensor_sub = torch.sub(tensor_mat, tensor_flatten)
    tensor_sub = tensor_sub.view([size[0], size[0], size[1]])
    tensor_sub = torch.abs(tensor_sub)
    if p == 0:
        tensor_sub = torch.pow(tensor_sub, p)
        dist = torch.sum(tensor_sub, dim=2)
        diag = torch.diag(dist)
        dist = torch.sub(dist, torch.diag(diag))
    elif p == 1:
        dist = torch.sum(tensor_sub, dim=2)
    else:
        tensor_sub = torch.pow(tensor_sub, p)
        dist = torch.sum(tensor_sub, dim=2)
        dist = torch.pow(dist, 1 / p)
    return dist


def torch_z_normalized(tensor: torch.Tensor, dim=0):
    """
    :param tensor: an 2D torch tensor
    :param dim:
        0 : normalize row data
        1 : normalize col data
    :return: Gaussian normalized tensor
    """
    mean = torch.mean(tensor, dim=1 - dim)
    std = torch.std(tensor, dim=1 - dim)
    if dim:
        tensor_sub_mean = torch.sub(tensor, mean)
        tensor_normalized = torch.div(tensor_sub_mean, std)
    else:
        size = mean.size()[0]
        tensor_sub_mean = torch.sub(tensor, mean.view([size, -1]))
        tensor_normalized = torch.div(tensor_sub_mean, std.view([size, -1]))
    return tensor_normalized


def exp_similarity(tensor: torch.Tensor, normalize=False):
    # def exp_similarity(tensor: torch.Tensor, sigma: torch.Tensor, normalize=True):
    """
    :param tensor: an torch tensor
    :param sigma: scale parameter
    :param normalize: normalize or not
    :return: exponential similarity
    """
    sigma = torch.tensor(2)
    if normalize:
        tensor = torch_z_normalized(tensor, dim=1)
    tensor_dist = torch_euclidean_dist(tensor, dim=0)
    exp_dist = torch.exp(-tensor_dist / (2 * torch.pow(sigma, 2)))
    return exp_dist


def jaccard_coef(tensor: torch.Tensor):
    """
    :param tensor: an torch tensor, 2D
    :return: jaccard coefficient
    """
    assert torch.all(tensor.le(1)) and torch.all(tensor.ge(0)), "Value must be 0 or 1"
    size = tensor.size()
    tensor_3d = (
        torch.flatten(tensor).repeat([size[0]]).view([size[0], size[0], size[1]])
    )
    ones = torch.ones(tensor_3d.size(), dtype=torch.float32, device=tensor.device)
    zeros = torch.zeros(tensor_3d.size(), dtype=torch.float32, device=tensor.device)
    tensor_3d = torch.add(tensor_3d, tensor.view([size[0], 1, size[1]]))
    intersection = torch.where(tensor_3d.eq(2), ones, zeros)
    union = torch.where(tensor_3d.eq(2), ones, tensor_3d)
    intersection = torch.sum(intersection, dim=2)
    union = torch.sum(union, dim=2)
    union = torch.where(union.eq(0), torch.add(union, 0.1), union)
    eye = torch.eye(union.size()[0], dtype=torch.float32, device=tensor.device)
    jaccard = torch.div(intersection, union)
    jaccard = torch.where(jaccard.eq(0), eye, jaccard)
    return jaccard


def full_kernel(exp_dist: torch.Tensor):
    """
    :param exp_dist: exponential similarity
    :return: full kernel
    """
    n = exp_dist.shape[0]
    ones = torch.ones(n, n, dtype=torch.float, device=exp_dist.device)
    one = torch.diag(torch.diag(ones))
    mask_diag = torch.mul(torch.sub(ones, one), exp_dist)
    mask_diag_sum = torch.sum(mask_diag, dim=1).view([n, -1])
    mask_diag = torch.div(mask_diag, 2 * mask_diag_sum)
    mask_diag = torch.add(mask_diag, 0.5 * one)
    return mask_diag


def sparse_kernel(exp_dist: torch.Tensor, k: int):
    """
    :param exp_dist: exponential similarity
    :param k: knn k
    :return: sparse kernel
    """
    n = exp_dist.shape[0]
    maxk = torch.topk(exp_dist, k, dim=1)
    mink_1 = torch.topk(exp_dist, n - k, dim=1, largest=False)
    index = torch.arange(n, device=exp_dist.device).view([n, -1])
    exp_dist[index, mink_1.indices] = 0
    knn_sum = torch.sum(maxk.values, dim=1).view([n, -1])
    exp_dist = torch.div(exp_dist, knn_sum)
    return exp_dist


def scale_sigmoid(tensor: torch.Tensor, alpha):
    """
    :param tensor: a torch tensor, range is [-1, 1]
    :param alpha: an scale parameter to sigmod
    :return: mapping tensor to [0, 1]
    """
    alpha = torch.tensor(alpha, dtype=torch.float32, device=tensor.device)
    output = torch.sigmoid(torch.mul(alpha, tensor))
    return output


def np_delete_value(arr: np.ndarray, obj: np.ndarray):
    """
    :param arr: 1-D vector, narray
    :param obj: 1-D vector, value that would be removed , narray
    :return: after removed vector
    """
    index = [np.where(x == arr)[0][0] for x in obj if x in arr]
    arr = np.delete(arr, index)
    return arr


def translate_result(tensor):
    """
    :param tensor: torch tensor or np.ndarray
    :return: pd.DataFrame
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    arr = tensor.reshape((1, -1))
    arr = pd.DataFrame(arr)
    return arr


def calculate_train_test_index(
    response: np.ndarray, pos_train_index: np.ndarray, pos_test_index: np.ndarray
):
    """
    :param response: response vector, np.ndarray
    :param pos_train_index: positive train index in response
    :param pos_test_index: positive test index in reaponse
    :return: train index and test index
    """
    neg_response_index = np.where(response == 0)[0]
    neg_test_index = np.random.choice(neg_response_index, pos_test_index.shape[0])
    neg_train_index = np_delete_value(neg_response_index, neg_test_index)
    test_index = np.hstack((pos_test_index, neg_test_index))
    train_index = np.hstack((pos_train_index, neg_train_index))
    return train_index, test_index


def dir_path(k=1):
    """
    :param k: 当前路径后退级数
    :return: 后退k级后的目录
    """
    fpath = os.path.realpath(__file__)
    dir_name = os.path.dirname(fpath)
    dir_name = dir_name.replace("\\", "/")
    p = len(dir_name) - 1
    while p > 0:
        if dir_name[p] == "/":
            k -= 1
            if k == 0:
                break
        p -= 1
    p += 1
    dir_name = dir_name[0:p]
    return dir_name


def extract_row_data(data: pd.DataFrame, row: int):
    """
    :param data: DataFrame
    :param row: row index, int
    :return: not nan data
    """
    data = np.array(data, dtype=np.float32)
    target = data[row, :]
    target = target[np.where(~np.isnan(target))[0]]
    return target


def transfer_data(data: pd.DataFrame, label: str):
    lenght = data.shape[0]
    target_label = np.array([label] * lenght)
    data["label"] = target_label
    return data


def link_data_frame(*data):
    """
    :param data: link DataFrame data
    :return: linked data
    """
    temp = pd.DataFrame()
    for i in data:
        temp = temp.append(i)
    return temp


def calculate_limit(*data, key):
    temp = pd.DataFrame()
    for value in data:
        temp = temp.append(value)
    max_value = temp[key].max() + 0.1
    min_value = temp[key].min() - 0.1
    return min_value, max_value


def delete_all_sub_str(string: str, sub: str, join_str=""):
    """
    :param string: long string, str
    :param sub: sub-string of string, str
    :param join_str: join string, str or None
    :return: after delete all sub string
    """
    string = string.split(sep=sub)
    string = np.array(string)
    string = join_str.join(np.delete(string, np.where(string == "")[0]))
    return string


def get_best_index(fname: str):
    """
    :param fname: file path and name
    :return: best index
    """
    file = open(fname, "r")
    string = file.readlines()
    string = "".join(string)
    string = string.replace("\n", "")
    temp = string.split(sep="accs")[0]
    temp = temp.split(sep=":")[1]
    aucs = delete_all_sub_str(string=temp, sub=" ", join_str=",")
    aucs = aucs.replace(",]", "]")
    aucs = np.array(eval(aucs))
    string = string.split(sep="accs")[1]
    avg_auc = string.split(sep="avg_aucs")[1]
    avg_auc = avg_auc.split(sep=" ")[0]
    avg_auc = float(avg_auc.split(sep=":")[1])
    abs_auc = np.abs(aucs - avg_auc)
    index = np.argmin(abs_auc)
    return index


def gather_color_code(*string):
    """
    :param string: colors, "blue", "orange", "green", "red", "purple", "brown", "pink", "grey", "yellow", "cyan"
    :return: colors code, list
    """
    color_str = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "grey",
        "yellow",
        "cyan",
    ]
    palette = sns.color_palette()
    color_map = dict(zip(color_str, palette))
    colors = [color_map[color] for color in string]
    return colors


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


# auc_ave: 0.94363 aupr_ave: 0.94358
def contrastive_loss(h1, h2, tau=0.6):
    sim_matrix = sim(h1, h2)
    f = lambda x: torch.exp(x / tau)
    matrix_t = f(sim_matrix)
    numerator = matrix_t.diag(dim=-1)
    denominator = torch.sum(matrix_t, dim=-1)
    loss = -torch.log(numerator / denominator).mean()
    return loss


def jaccard_coef7(tensor: torch.Tensor):
    """
    :param tensor: an torch tensor, 2D
    :return: jaccard coefficient
    """
    assert torch.all(tensor.le(1)) and torch.all(tensor.ge(0)), "Value must be 0 or 1"
    size = tensor.size()
    tensor_3d = (
        torch.flatten(tensor).repeat([size[0]]).view([size[0], size[0], size[1]])
    )
    ones = torch.ones_like(tensor_3d)
    zeros = torch.zeros_like(tensor_3d)
    tensor_3d = torch.add(tensor_3d, tensor.view([size[0], 1, size[1]]))
    intersection = torch.where(tensor_3d.eq(2), ones, zeros)
    union = torch.where(tensor_3d.eq(2), ones, tensor_3d)
    intersection = torch.sum(intersection, dim=2)
    union = torch.sum(union, dim=2)
    union = torch.where(union.eq(0), torch.add(union, 0.1), union)
    eye = torch.eye(union.size()[0], dtype=tensor.dtype, device=tensor.device)
    jaccard = torch.div(intersection, union)
    jaccard = torch.where(jaccard.eq(0), eye, jaccard)
    return jaccard


def gaussian_normalization(x: torch.Tensor):
    mean = torch.mean(x, dim=0)
    std = torch.std(x, dim=0)
    return torch.div(torch.sub(x, mean.view([1, -1])), std.view([1, -1]))


def euclidean_distance(x: torch.Tensor):
    x = torch.mm(x, x.t())
    diag = torch.diag(x)
    diag = torch.add(diag.view([-1, 1]), diag.view([1, -1]))
    x = torch.sub(diag, torch.mul(x, 2.0))
    return torch.sqrt(x)


def exponent_similarity(dist: torch.Tensor, mu: float):
    denominate = torch.square(torch.tensor(mu, dtype=dist.dtype, device=dist.device))
    dist = torch.div(dist, torch.mul(2, denominate))
    dist = torch.exp(-dist)
    return dist


def calculate_gene_exponent_similarity7(x: torch.Tensor, mu: float):
    """
    Calculate the gene exponent similarity
    :param x: gene feature
    :param mu: scale parameter
    :return: cell line gene exponent similarity
    """
    x = gaussian_normalization(x=x)
    x = euclidean_distance(x=x)
    return exponent_similarity(dist=x, mu=mu)


def gaussian_kernel_matrix(X, sigma=1):
    # 获取样本数量
    n_samples = X.shape[0]

    # 初始化相似性矩阵
    similarity_matrix = np.zeros((n_samples, n_samples))

    # 计算高斯核相似性
    for i in range(n_samples):
        for j in range(n_samples):
            x = X[i, :]
            y = X[j, :]
            similarity_matrix[i, j] = np.exp(-np.sum((x - y) ** 2) / (2 * sigma**2))

    return similarity_matrix


def generate_mask(row, column, mask_ratio):
    # 1 -- leave   0 -- drop
    arr_mask_ratio = np.random.uniform(0, 1, size=(row, column))
    arr_mask = np.ma.masked_array(
        arr_mask_ratio, mask=(arr_mask_ratio < mask_ratio)
    ).filled(0)
    arr_mask = np.ma.masked_array(arr_mask, mask=(arr_mask >= mask_ratio)).filled(1)
    return arr_mask


def save_dataset(test_mask, null_mask, res):
    vv = 1 - (test_mask + null_mask)
    vv = to_coo_matrix(vv)
    row = pd.DataFrame(vv.row)
    col = pd.DataFrame(vv.col)
    data = pd.DataFrame(res[vv.row, vv.col])
    v = pd.concat([row, col, data], axis=1)
    v.columns = ["cells", "drugs", "labels"]

    tt = to_coo_matrix(test_mask)
    row = pd.DataFrame(tt.row)
    col = pd.DataFrame(tt.col)
    data = pd.DataFrame(res[tt.row, tt.col])
    t = pd.concat([row, col, data], axis=1)
    t.columns = ["cells", "drugs", "labels"]

    return v, t
