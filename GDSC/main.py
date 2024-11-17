import argparse
import numpy as np
import pandas as pd
from load_data import load_data
from models import hrlcdr, Optimizer_mul
from sklearn.model_selection import KFold
from sampler import RandomSampler_yz
from myutils import *
import logging
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#================== 声明 ==========================

parser = argparse.ArgumentParser(description="Run HRLCDR")
parser.add_argument('-device', type=str, default="cpu", help='cuda:number or cpu')
parser.add_argument('-data', type=str, default='gdsc', help='Dataset{gdsc or ccle}')
parser.add_argument('--lr', type=float,default=0.001,
                    help="the learning rate")
parser.add_argument('--wd', type=float,default=1e-5,
                    help="the weight decay for l2 normalizaton")
parser.add_argument('--gamma', type=float,default=8.7,
                    help="the scale for sigmod")
parser.add_argument('--epochs', type=float,default=1000,
                    help="the epochs for model")
args = parser.parse_args()

logging.basicConfig(
    format='%(message)s',
    filename='mylog.log',
    level=logging.INFO
)

#===================== data prepare ================

res_ic50, res_binary, drug_finger, exprs, null_mask, pos_num, args = load_data(args)

# Dynamically set the device based on the argument
if args.device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device(args.device)

cell_sim_f = calculate_gene_exponent_similarity7(x=torch.from_numpy(exprs).to(dtype=torch.float32, device=device), mu=3)
drug_sim_f = jaccard_coef7(tensor=torch.from_numpy(drug_finger).to(dtype=torch.float32, device=device))


cell_sim_f = k_near_graph(cell_sim_f, 11, device)
drug_sim_f = k_near_graph(drug_sim_f, 11, device)


k = 5
n_kfolds = 5

true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()

aucs = []
auprs = []
sed = 100
#==================================dataset split==============================
np.random.seed(sed)
a = np.arange(pos_num)
b = np.random.choice(a, size=2085, replace=False)
ind = np.zeros(20851, dtype=bool)
ind[b] = True
a = a[~ind]
pos_adj_mat = null_mask + res_binary
neg_adj_mat = sp.coo_matrix(np.abs(pos_adj_mat - np.array(1)))
all_row = neg_adj_mat.row
all_col = neg_adj_mat.col
all_data = neg_adj_mat.data
index = np.arange(all_data.shape[0])
c = np.random.choice(index, 2085, replace=False)
kfold = KFold(n_splits=k, shuffle=True, random_state=0)

for n_kfold in range(n_kfolds):
    for train_index, val_index in kfold.split(a):
        train_i = np.zeros(len(a), dtype=bool)
        train_i[train_index] = True
        val_i = np.zeros(len(a), dtype=bool)
        val_i[val_index] = True
        sampler = RandomSampler_yz(res_ic50, res_binary, a[train_i], a[val_i], null_mask,b,c)

        ttt = np.array(sampler.test_mask).astype(float)
        ttt2 = np.array(sampler.val_mask).astype(float)
        tra = (1 - ttt - null_mask -ttt2)
        cv = np.multiply(res_binary,tra)
        cv = torch.from_numpy(cv).to(dtype=torch.float32, device=device)
        A1, A2, A3, A4,A5,A6 = hyper_graph(drug_sim_f, cv.T, cell_sim_f)  # drug hyper
        B1, B2, B3, B4,B5,B6 = hyper_graph(cell_sim_f, cv, drug_sim_f)  # cell hyper

        B4 = torch.mul(B4, (1.0/B4.sum(axis=1).reshape(-1,1)))
        A6 = torch.mul(A6, (1.0/A6.sum(axis=1).reshape(-1,1)))


        model = hrlcdr(adj_mat=sampler.train_data,  cell_exprs=exprs, drug_finger=drug_finger,
                        gamma=args.gamma,drug_hyper = A6, cell_hyper = B4,device=device).to(device)

        opt = Optimizer_mul(model, sampler.train_ic50, sampler.val_ic50, sampler.train_data, sampler.val_data, sampler.val_mask, sampler.train_mask,
                            sampler.test_data,sampler.test_mask,sampler.test_ic50,
                roc_auc,ap_score, lr=args.lr, wd=args.wd, epochs=args.epochs, device=device).to(device)

        auc,aupr, true_data, best_predict = opt()
        true_datas = pd.concat([true_datas, translate_result(true_data)], ignore_index=True)
        predict_datas = pd.concat([predict_datas, translate_result(best_predict)], ignore_index=True)
        auc, aupr, _, _, _ = evaluate_all(true_data, best_predict)
        aucs.append(auc)
        auprs.append(aupr)
        print('current auc and time :',np.mean(aucs),len(aucs))
        print('current aupr and time :',np.mean(auprs),len(auprs))

print('auc : ',np.mean(aucs))
print('aupr : ',np.mean(auprs))
print(len(aucs))
print(len(auprs))
logging.info(f"final_auc:{np.mean(aucs):.4f},var:{np.var(aucs)},final_aupr:{np.mean(auprs):.4f},var:{np.var(auprs)}")
cv_set, test_set = save_dataset(sampler.test_mask,null_mask,res_binary)
cv_set.to_csv("Data/5-fold_CV.csv",index=False)
test_set.to_csv("Data/testset.csv",index=False)

pd.DataFrame(true_datas).to_csv("result_data/true_data.csv")
pd.DataFrame(predict_datas).to_csv("result_data/predict_data.csv")
