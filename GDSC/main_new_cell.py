# coding: utf-8
import sys

sys.path.append("./.")
import argparse
import copy
import logging

import numpy as np
import pandas as pd
from load_data import load_data
from myutils import *
from sampler import RandomSampler
from sklearn.model_selection import KFold

from HRLCDR.GDSC.HRLCDR_New import HRLCDR_new

parser = argparse.ArgumentParser(description="Run HRLCDR")
parser.add_argument("-device", type=str, default="cuda:0", help="cuda:number or cpu")
parser.add_argument("-data", type=str, default="gdsc", help="Dataset{gdsc or ccle}")
parser.add_argument("--lr", type=float, default=0.05, help="the learning rate")
parser.add_argument(
    "--wd", type=float, default=1e-5, help="the weight decay for l2 normalizaton"
)
parser.add_argument("--gamma", type=float, default=8.7, help="the scale for sigmod")
parser.add_argument("--epochs", type=float, default=2000, help="the epochs for model")
parser.add_argument("--tol", type=float, default=100, help="early stop count for model")

args = parser.parse_args()


logging.basicConfig(format="%(message)s", filename="mylog.log", level=logging.INFO)

# ===================== data prepare ================

res_ic50, cell_drug, drug_finger, exprs, null_mask1, pos_num, args = load_data(args)

ori_null_mask = copy.deepcopy(null_mask1)
cell_sim_fo = calculate_gene_exponent_similarity7(
    x=torch.from_numpy(exprs).to(dtype=torch.float32, device="cuda:0"), mu=3
)
drug_sim_fo = jaccard_coef7(
    tensor=torch.from_numpy(drug_finger).to(dtype=torch.float32, device="cuda:0")
)

cell_sim_f = k_near_graph(cell_sim_fo, 11)
drug_sim_f = k_near_graph(drug_sim_fo, 11)

# ====================================dataset=========================================

kf = KFold(n_splits=5, shuffle=True, random_state=27)
kfold = KFold(n_splits=5, shuffle=True, random_state=27)
x, y = cell_drug.shape

t_dim = 0
final_auc_d = []
final_aupr_d = []


for train_index, test_index in kf.split(np.arange(x)):
    null_mask = copy.deepcopy(null_mask1)
    train_p = copy.deepcopy(cell_drug)
    train_n = copy.deepcopy(cell_drug + null_mask - 1)
    if t_dim == 0:
        # row
        train_p[test_index, :] = 0
        train_n[test_index, :] = 0
    else:
        # col
        args.lr = 0.05
        args.tol = 100
        train_p[:, test_index] = 0
        train_n[:, test_index] = 0
    train_p = sp.coo_matrix(train_p)
    train_n = sp.coo_matrix(train_n)
    train_pos = list(zip(train_p.row, train_p.col, train_p.data))
    train_neg = list(zip(train_n.row, train_n.col, train_n.data + 1))
    train_neg.extend(train_pos)
    train_samples = np.array(train_neg)
    test_p = copy.deepcopy(cell_drug)
    test_n = copy.deepcopy(cell_drug + null_mask - 1)
    if t_dim == 0:
        # row
        test_p[train_index, :] = 0
        test_n[train_index, :] = 0
    else:
        # col
        test_p[:, train_index] = 0
        test_n[:, train_index] = 0

    test_p = sp.coo_matrix(test_p)
    test_n = sp.coo_matrix(test_n)
    test_pos = list(zip(test_p.row, test_p.col, test_p.data))
    test_neg = list(zip(test_n.row, test_n.col, test_n.data + 1))
    test_neg.extend(test_pos)
    test_samples = np.array(test_neg)
    new_A = update_Adjacency_matrix(cell_drug, test_samples)
    if t_dim == 0:
        # row
        null_mask[test_index, :] = 1
    else:
        # col
        null_mask[:, test_index] = 1
    # ===================== negtive sampler ====================

    neg_adj_mat = copy.deepcopy(cell_drug + ori_null_mask)
    neg_adj_mat = np.abs(neg_adj_mat - np.array(1))
    if t_dim == 0:
        # row
        neg_adj_mat[train_index, :] = 0
    else:
        # col
        neg_adj_mat[:, train_index] = 0

    neg_adj_mat = sp.coo_matrix(neg_adj_mat)
    pos_adj_mat = copy.deepcopy(cell_drug)
    if t_dim == 0:
        # row
        pos_adj_mat[train_index, :] = 0
    else:
        # col
        pos_adj_mat[:, train_index] = 0

    all_row = neg_adj_mat.row
    all_col = neg_adj_mat.col
    all_data = neg_adj_mat.data
    index = np.arange(all_data.shape[0])
    # 采样负测试集
    test_n = int(pos_adj_mat.sum())
    test_neg_index = np.random.choice(index, test_n, replace=False)
    test_row = all_row[test_neg_index]
    test_col = all_col[test_neg_index]
    test_data = all_data[test_neg_index]
    test = sp.coo_matrix((test_data, (test_row, test_col)), shape=cell_drug.shape)
    train = sp.coo_matrix(pos_adj_mat)
    banlence_mask = mask(train, test, dtype=bool)  # independent test set

    # ==========================================================================

    for train_ind, test_ind in kfold.split(np.arange(sum(sum(new_A)))):

        true_data_s = pd.DataFrame()
        predict_data_s = pd.DataFrame()
        sam = RandomSampler(
            new_A, train_ind, test_ind, null_mask
        )  #  cross-validation set
        ttt = np.array(sam.test_mask).astype(float)
        ttt2 = np.array(banlence_mask).astype(float)
        tra = 1 - ttt - null_mask - ttt2
        if t_dim == 0:
            # row
            tra[train_index, :] = 0
        else:
            # col
            tra[:, train_index] = 0
        cv = np.multiply(cell_drug, tra)
        cv = torch.from_numpy(cv).to(dtype=torch.float32, device="cuda:0")
        A1, A2, A3, A4, A5, A6 = hyper_graph(drug_sim_f, cv.T, cell_sim_f)  # drug hyper
        B1, B2, B3, B4, B5, B6 = hyper_graph(cell_sim_f, cv, drug_sim_f)  # cell hyper

        auc, aupr, true_data, predict_data = HRLCDR_new(
            cell_exprs=exprs,
            drug_finger=drug_finger,
            sam=sam,
            test_da=cell_drug,
            test_ma=banlence_mask,
            args=args,
            cell_hyper=B4,
            drug_hyper=A6,
            A1=A1,
            A2=A2,
            A3=A3,
            B1=B1,
            B2=B2,
            B3=B3,
            device=args.device,
            alp=0.9,
            bet=0.1,
        )

        true_data_s = pd.concat([true_data_s, translate_result(true_data)])
        predict_data_s = pd.concat([predict_data_s, translate_result(predict_data)])

        final_auc_d.append(auc)
        final_aupr_d.append(aupr)
        print("creent auc :", np.mean(final_auc_d))


if t_dim == 0:
    print("new_cell_auc:", np.mean(final_auc_d), np.var(final_auc_d))
    print("new_cell_aupr:", np.mean(final_aupr_d), np.var(final_aupr_d))

else:
    print("new_drug_auc:", np.mean(final_auc_d), np.var(final_auc_d))
    print("new_drug_aupr:", np.mean(final_aupr_d), np.var(final_aupr_d))

logging.info(
    f"final_auc:{np.mean(final_auc_d):.4f},var:{np.var(final_auc_d)},final_aupr:{np.mean(final_aupr_d):.4f},var:{np.var(final_aupr_d)}"
)
