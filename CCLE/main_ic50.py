import argparse
import logging
import warnings

import numpy as np
import pandas as pd
from load_data import load_data
from models import Optimizer_mul_ic50, hrlcdr_ic50
from myutils import *
from sampler import RandomSampler_yz
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ================== 声明 ==========================

parser = argparse.ArgumentParser(description="Run HRLCDR")
parser.add_argument("-device", type=str, default="cuda:0", help="cuda:number or cpu")
parser.add_argument("-data", type=str, default="ccle", help="Dataset{gdsc or ccle}")
parser.add_argument("--lr", type=float, default=0.001, help="the learning rate")
parser.add_argument(
    "--wd", type=float, default=1e-5, help="the weight decay for l2 normalizaton"
)
parser.add_argument("--gamma", type=float, default=8.7, help="the scale for sigmod")
parser.add_argument("--epochs", type=float, default=3000, help="the epochs for model")
args = parser.parse_args()

logging.basicConfig(format="%(message)s", filename="mylog.log", level=logging.INFO)

# ===================== data prepare ================

res_ic50, res_binary, drug_finger, exprs, null_mask, pos_num, args = load_data(args)
cell_sim_f = calculate_gene_exponent_similarity7(
    x=torch.from_numpy(exprs).to(dtype=torch.float32, device="cuda:0"), mu=3
)
drug_sim_f = jaccard_coef7(
    tensor=torch.from_numpy(drug_finger).to(dtype=torch.float32, device="cuda:0")
)

cell_sim_f = k_near_graph(cell_sim_f, 11)
drug_sim_f = k_near_graph(drug_sim_f, 11)


k = 5
n_kfolds = 5

true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()

pccs = []
sccs = []
rmses = []
sed = 2023
# ==================================dataset split==============================
np.random.seed(sed)
a = np.arange(pos_num)
b = np.random.choice(a, size=169, replace=False)
ind = np.zeros(1696, dtype=bool)
ind[b] = True
a = a[~ind]
pos_adj_mat = null_mask + res_binary
neg_adj_mat = sp.coo_matrix(np.abs(pos_adj_mat - np.array(1)))
all_row = neg_adj_mat.row
all_col = neg_adj_mat.col
all_data = neg_adj_mat.data
index = np.arange(all_data.shape[0])
c = np.random.choice(index, 169, replace=False)
kfold = KFold(n_splits=k, shuffle=True, random_state=2023)


for n_kfold in range(n_kfolds):
    for train_index, val_index in kfold.split(a):
        train_i = np.zeros(len(a), dtype=bool)
        train_i[train_index] = True
        val_i = np.zeros(len(a), dtype=bool)
        val_i[val_index] = True
        sampler = RandomSampler_yz(
            res_ic50, res_binary, a[train_i], a[val_i], null_mask, b, c
        )
        ttt = np.array(sampler.test_mask).astype(float)
        ttt2 = np.array(sampler.val_mask).astype(float)
        tra = 1 - ttt - null_mask - ttt2
        cv = np.multiply(res_binary, tra)
        cv = torch.from_numpy(cv).to(dtype=torch.float32, device="cuda:0")
        A1, A2, A3, A4, A5, A6 = hyper_graph(drug_sim_f, cv.T, cell_sim_f)  # drug hyper
        B1, B2, B3, B4, B5, B6 = hyper_graph(cell_sim_f, cv, drug_sim_f)  # cell hyper
        B4 = torch.mul(B4, (1.0 / B4.sum(axis=1).reshape(-1, 1)))
        A6 = torch.mul(A6, (1.0 / A6.sum(axis=1).reshape(-1, 1)))  #
        # ================================== split end ====================================

        model = hrlcdr_ic50(
            adj_mat=sampler.train_data,
            cell_exprs=exprs,
            drug_finger=drug_finger,
            gamma=args.gamma,
            drug_hyper=A6,
            cell_hyper=B4,
            device=args.device,
        ).to(args.device)

        opt = Optimizer_mul_ic50(
            model,
            sampler.train_ic50,
            sampler.val_ic50,
            sampler.train_data,
            sampler.val_data,
            sampler.val_mask,
            sampler.train_mask,
            sampler.test_data,
            sampler.test_mask,
            sampler.test_ic50,
            pcc_score,
            sc_score,
            rmse_score,
            lr=args.lr,
            wd=args.wd,
            epochs=args.epochs,
            device=args.device,
        ).to(args.device)

        pcc, scc, rmse, true_data, best_predict = opt()
        true_datas = true_datas.append(translate_result(true_data))
        predict_datas = predict_datas.append(translate_result(best_predict))
        pccs.append(pcc)
        print("current pcc and time :", np.mean(pccs), len(pccs))
        sccs.append(scc)
        rmses.append(rmse)

print("pccs : ", np.mean(pccs))
print("sccs : ", np.mean(sccs))
print("rmses : ", np.mean(rmses))
cv_set, test_set = save_dataset(sampler.test_mask, null_mask, res_ic50)
cv_set.to_csv("./HRLCDR/CCLE/Data/5-fold_CV_ic50.csv", index=False)
test_set.to_csv("./HRLCDR/CCLE/Data/testset_ic50.csv", index=False)
pd.DataFrame(true_datas).to_csv("./HRLCDR/CCLE/result_data/true_data_ic50.csv")
pd.DataFrame(predict_datas).to_csv("./HRLCDR/CCLE/result_data/predict_data_ic50.csv")

logging.info(
    f"final_pcc:{np.mean(pccs):.4f},var:{np.var(pccs)},final_sccs:{np.mean(sccs):.4f},var:{np.var(sccs)}"
    f"final_rmses:{np.mean(rmses):.4f},var:{np.var(rmses)}"
)
