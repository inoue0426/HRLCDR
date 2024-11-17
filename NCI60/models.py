from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim
from myutils import *


class ConstructAdjMatrix(nn.Module, ABC):
    def __init__(self, original_adj_mat, device="cpu"):
        super(ConstructAdjMatrix, self).__init__()
        self.adj = original_adj_mat.to(device)
        self.device = device

    def forward(self):
        d_x = torch.diag(torch.pow(torch.sum(self.adj, dim=1) + 1, -0.5))
        d_y = torch.diag(torch.pow(torch.sum(self.adj, dim=0) + 1, -0.5))

        agg_cell_lp = torch.mm(torch.mm(d_x, self.adj), d_y)
        agg_drug_lp = torch.mm(torch.mm(d_y, self.adj.T), d_x)

        d_c = torch.pow(torch.sum(self.adj, dim=1) + 1, -1)
        self_cell_lp = torch.diag(torch.add(d_c, 1))
        d_d = torch.pow(torch.sum(self.adj, dim=0) + 1, -1)
        self_drug_lp = torch.diag(torch.add(d_d, 1))
        return agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp


class LoadFeature(nn.Module, ABC):
    def __init__(self, cell_exprs, drug_finger, device="cpu"):
        super(LoadFeature, self).__init__()
        cell_exprs = torch.from_numpy(cell_exprs).to(device)  # norm
        self.cell_feat = torch_z_normalized(cell_exprs, dim=1).to(device)  # norm
        # self.cell_feat = torch.from_numpy(cell_exprs).to(device).to(dtype=torch.float) # nan norm
        self.drug_feat = torch.from_numpy(drug_finger).to(device).to(dtype=torch.float)

    def forward(self):
        cell_feat = self.cell_feat
        drug_feat = self.drug_feat
        return cell_feat, drug_feat


class GDecoder(nn.Module, ABC):
    def __init__(self, gamma):
        super(GDecoder, self).__init__()
        self.gamma = gamma

    def forward(self, cell_emb, drug_emb):
        Corr = torch_corr_x_y(cell_emb, drug_emb)
        output = scale_sigmoid(Corr, alpha=self.gamma)
        return output


class GDecoder_regression(nn.Module, ABC):
    def __init__(self, gamma):
        super(GDecoder_regression, self).__init__()
        self.gamma = gamma

    def forward(self, cell_emb, drug_emb):
        output = torch.mul(torch_corr_x_y(cell_emb, drug_emb), self.gamma)
        return output


class Hyper_Agg(nn.Module):
    def __init__(
        self,
        adj_mat,
        adj_mat_T,
        self_c,
        self_d,
        cell_hyper,
        drug_hyper,
        alp,
        bet,
        device="cpu",
    ):
        super(Hyper_Agg, self).__init__()

        self.adj_mat = adj_mat
        self.adj_mat_T = adj_mat_T
        self.lp_c = self_c
        self.lp_d = self_d

        self.cell_hyper = cell_hyper
        self.drug_hyper = drug_hyper

        A4H = torch.diag(torch.pow(torch.sum(drug_hyper, dim=1) + 1, -1)).to(device)
        B4H = torch.diag(torch.pow(torch.sum(cell_hyper, dim=1) + 1, -1)).to(device)

        # N = 60, Num of cell lines.
        self.chaotuHC = torch.eye(60).to(device) - torch.mm(
            torch.mm(B4H, cell_hyper), B4H
        )

        # M = 952, Num of drugs.
        self.chaotuHD = torch.eye(952).to(device) - torch.mm(
            torch.mm(A4H, drug_hyper), A4H
        )

        self.in_dim = 2040
        self.out_dim = 2040

        self.agg_cs = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.agg_c1 = nn.Linear(self.in_dim, self.out_dim, bias=True)

        self.agg_ds = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.agg_d1 = nn.Linear(self.in_dim, self.out_dim, bias=True)

        self.agg_b4 = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.agg_a4 = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.agg_b42 = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.agg_a42 = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.agg_b4H = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.agg_a4H = nn.Linear(self.in_dim, self.out_dim, bias=True)

        self.al = alp
        self.bt = bet

    def forward(self, exp, fig):
        alp = self.al
        bet = self.bt

        cells_1 = self.agg_cs(torch.mm(self.lp_c, exp))
        cells_r = torch.mul(cells_1, exp)
        cells = bet * cells_1 + alp * cells_r

        cell1_1 = self.agg_c1(torch.mm(self.adj_mat, fig))
        cell1 = alp * torch.mul(cell1_1, exp) + bet * cell1_1

        drugs_1 = self.agg_ds(torch.mm(self.lp_d, fig))
        drugs_r = torch.mul(drugs_1, fig)
        drugs = bet * drugs_1 + alp * drugs_r

        drug1_1 = self.agg_d1(torch.mm(self.adj_mat_T, exp))
        drug1 = alp * torch.mul(drug1_1, fig) + bet * drug1_1

        cell4 = self.agg_b4(torch.mm(self.cell_hyper, exp))
        cell5 = self.agg_b4H(torch.mm(self.chaotuHC, exp))
        cell42 = alp * torch.mul(cell4, exp) + bet * cell4
        cell6 = alp * torch.mul(cell5, exp) + bet * cell5

        drug4 = self.agg_a4(torch.mm(self.drug_hyper, fig))
        drug42 = alp * torch.mul(drug4, fig) + bet * drug4
        drug5 = self.agg_a4H(torch.mm(self.chaotuHD, fig))
        drug6 = alp * torch.mul(drug5, fig) + bet * drug5

        cellagg = fun.relu(cells + cell1 + cell6 + cell42)
        drugagg = fun.relu(drugs + drug1 + drug6 + drug42)

        return cellagg, drugagg


class hrlcdr(nn.Module, ABC):
    def __init__(
        self,
        adj_mat,
        cell_exprs,
        drug_finger,
        gamma,
        drug_hyper,
        cell_hyper,
        device="cpu",
    ):
        super(hrlcdr, self).__init__()

        construct_adj_matrix = ConstructAdjMatrix(adj_mat, device=device)
        loadfeat = LoadFeature(cell_exprs, drug_finger, device=device)
        agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp = construct_adj_matrix()
        cell_feat, drug_feat = loadfeat()
        self.cexp = cell_feat
        self.figprint = drug_feat

        self.ldd = nn.Linear(self.figprint.shape[1], 2040, bias=True).to(device)
        self.lcc = nn.Linear(cell_exprs.shape[1], 2040, bias=True).to(device)

        self.agg_h = Hyper_Agg(
            adj_mat=agg_cell_lp,
            adj_mat_T=agg_drug_lp,
            self_c=self_cell_lp,
            self_d=self_drug_lp,
            cell_hyper=cell_hyper,
            drug_hyper=drug_hyper,
            alp=1,
            bet=1,
        ).to(device)
        self.decoder = GDecoder(gamma=gamma)

        self.bnd1 = nn.BatchNorm1d(2040)
        self.bnc1 = nn.BatchNorm1d(2040)

    def forward(self):

        drug_x = self.bnd1(self.ldd(self.figprint))
        cell_emb_3_t = self.bnc1(self.lcc(self.cexp))
        cell_emb_3_t_out, drug_x_out = self.agg_h(cell_emb_3_t, drug_x)
        output = self.decoder(cell_emb_3_t_out, drug_x_out)

        return output


class hrlcdr_ic50(nn.Module, ABC):
    def __init__(
        self,
        adj_mat,
        cell_exprs,
        drug_finger,
        gamma,
        drug_hyper,
        cell_hyper,
        device="cpu",
    ):
        super(hrlcdr_ic50, self).__init__()

        construct_adj_matrix = ConstructAdjMatrix(adj_mat, device=device)
        loadfeat = LoadFeature(cell_exprs, drug_finger, device=device)
        agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp = construct_adj_matrix()
        cell_feat, drug_feat = loadfeat()
        self.cexp = cell_feat
        self.figprint = drug_feat

        self.ldd = nn.Linear(self.figprint.shape[1], 2040, bias=True).to(device)
        self.lcc = nn.Linear(cell_exprs.shape[1], 2040, bias=True).to(device)
        self.agg_h = Hyper_Agg(
            adj_mat=agg_cell_lp,
            adj_mat_T=agg_drug_lp,
            self_c=self_cell_lp,
            self_d=self_drug_lp,
            cell_hyper=cell_hyper,
            drug_hyper=drug_hyper,
            alp=1,
            bet=1,
        ).to(device)

        self.bnd1 = nn.BatchNorm1d(2040)
        self.bnc1 = nn.BatchNorm1d(2040)
        self.decoder = GDecoder_regression(gamma=gamma)

    def forward(self):

        drug_x = self.bnd1(self.ldd(self.figprint))
        cell_emb_3_t = self.bnc1(self.lcc(self.cexp))
        cell_emb_3_t_out, drug_x_out = self.agg_h(cell_emb_3_t, drug_x)
        output = self.decoder(cell_emb_3_t_out, drug_x_out)

        return output


class Optimizer_mul(nn.Module):
    def __init__(
        self,
        model,
        train_ic50,
        test_ic50,
        train_data,
        test_data,
        test_mask,
        train_mask,
        independ_data,
        ind_mask,
        ind_ic50,
        evaluate_fun,
        evluate_fun2,
        lr=0.01,
        wd=1e-05,
        epochs=200,
        test_freq=20,
        device="cpu",
    ):
        super(Optimizer_mul, self).__init__()
        self.model = model.to(device)

        self.train_ic50 = train_ic50.to(device)
        self.test_ic50 = test_ic50.to(device)
        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.train_mask = train_mask.to(device)
        self.test_mask = test_mask.to(device)

        self.ind_ic50 = ind_ic50.to(device)
        self.ind_data = independ_data.to(device)
        self.ind_mask = ind_mask.to(device)

        self.evaluate_fun = evaluate_fun
        self.evaluate_fun2 = evluate_fun2
        self.lr = lr
        self.wd = wd

        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )

    def forward(self):
        best_auc = 0
        tole = 0

        true_data = torch.masked_select(self.test_data, self.test_mask)
        ind_true_data = torch.masked_select(self.ind_data, self.ind_mask)

        for epoch in torch.arange(self.epochs + 1):

            predict_data = self.model()

            loss = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ind_data_masked = torch.masked_select(predict_data, self.ind_mask)
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            auc = self.evaluate_fun(true_data, predict_data_masked)

            if auc > best_auc:
                best_auc = auc
                best_predict = torch.masked_select(predict_data, self.ind_mask)
                ind_auc = self.evaluate_fun(ind_true_data, ind_data_masked)
                ind_aupr = self.evaluate_fun2(ind_true_data, ind_data_masked)
                tole = 0
            else:
                tole += 1

            if epoch % self.test_freq == 0:
                print(
                    "epoch:%4d" % epoch.item(),
                    "loss:%.6f" % loss.item(),
                    "auc:%.4f" % auc,
                    "ind_auc:%.4f" % ind_auc,
                )

        print("Fit finished.")

        return ind_auc, ind_aupr, ind_true_data, best_predict


class hrlcdr_new(nn.Module, ABC):
    def __init__(
        self,
        adj_mat,
        cell_exprs,
        drug_finger,
        gamma,
        drug_hyper,
        cell_hyper,
        device="cpu",
        alp=1,
        bet=1,
    ):
        super(hrlcdr_new, self).__init__()

        construct_adj_matrix = ConstructAdjMatrix(adj_mat, device=device)
        loadfeat = LoadFeature(cell_exprs, drug_finger, device=device)
        agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp = construct_adj_matrix()
        cell_feat, drug_feat = loadfeat()
        self.cexp = cell_feat
        self.figprint = drug_feat

        dim = 2040
        self.ldd = nn.Linear(self.figprint.shape[1], dim, bias=True).to(device)
        self.lcc = nn.Linear(self.cexp.shape[1], dim, bias=True).to(device)

        self.agg_h = Hyper_Agg(
            adj_mat=agg_cell_lp,
            adj_mat_T=agg_drug_lp,
            self_c=self_cell_lp,
            self_d=self_drug_lp,
            cell_hyper=cell_hyper,
            drug_hyper=drug_hyper,
            alp=alp,
            bet=bet,
        ).to(device)
        self.decoder = GDecoder(gamma=gamma)

        self.bnd1 = nn.BatchNorm1d(dim)
        self.bnc1 = nn.BatchNorm1d(dim)

    def forward(self):

        drug_x = self.bnd1(self.ldd(self.figprint))

        cell_emb_3_t = self.bnc1(self.lcc(self.cexp))

        cell_emb_3_t_out, drug_x_out = self.agg_h(cell_emb_3_t, drug_x)
        output = self.decoder(cell_emb_3_t_out, drug_x_out)

        return output


class Optimizer_new(nn.Module):
    def __init__(
        self,
        model,
        train_data,
        test_data,
        test_mask,
        train_mask,
        ind_data,
        ind_mask,
        evaluate_fun,
        evaluate_fun2,
        lr=0.001,
        wd=1e-05,
        epochs=200,
        tol=50,
        test_freq=20,
        device="cpu",
    ):  #
        super(Optimizer_new, self).__init__()
        self.model = model.to(device)

        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.train_mask = train_mask.to(device)
        self.test_mask = test_mask.to(device)

        self.evaluate_fun = evaluate_fun
        self.evaluate_fun2 = evaluate_fun2

        self.lr = lr
        self.early_stop_count = tol
        self.wd = wd

        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )

        self.ind_data = torch.from_numpy(ind_data).float().to(device)
        self.ind_mask = ind_mask.to(device)

    def forward(self):
        best_auc = 0
        best_aupr = 0
        tol = 0
        tol_auc = 0
        true_data = torch.masked_select(self.test_data, self.test_mask)
        train_da = torch.masked_select(self.ind_data, self.ind_mask)

        for epoch in torch.arange(self.epochs + 1):

            predict_data = self.model()
            loss_binary = cross_entropy_loss(
                self.train_data, predict_data, self.train_mask
            )
            loss = loss_binary
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            predict_ind_masked = torch.masked_select(predict_data, self.ind_mask)
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            train_auc = self.evaluate_fun(train_da, predict_ind_masked)
            train_aupr = self.evaluate_fun2(train_da, predict_ind_masked)
            auc = self.evaluate_fun(true_data, predict_data_masked)

            if tol != self.early_stop_count:
                if auc > tol_auc:
                    tol = 0
                    tol_auc = auc
                    best_auc = train_auc

                    best_predict = predict_ind_masked
                    best_aupr = train_aupr
                else:
                    tol += 1
            else:
                break
            if epoch % self.test_freq == 0:
                print(
                    "epoch:%4d" % epoch.item(),
                    "loss:%.4f" % loss.item(),
                    "auc:%.4f" % best_auc,
                    "aupr:%.4f" % best_aupr,
                    "yanzheng:%.4f" % auc,
                )

        print("Fit finished.")
        return best_auc, best_aupr, train_da, best_predict  # true_data


class Optimizer_mul_ic50(nn.Module):
    def __init__(
        self,
        model,
        train_ic50,
        test_ic50,
        train_data,
        test_data,
        test_mask,
        train_mask,
        independ_data,
        ind_mask,
        ind_ic50,
        evaluate_fun,
        sc_score,
        rmse_score,
        lr=0.01,
        wd=1e-05,
        epochs=200,
        test_freq=20,
        device="cpu",
    ):
        super(Optimizer_mul_ic50, self).__init__()
        self.model = model.to(device)

        self.train_ic50 = train_ic50.to(device)

        self.test_ic50 = test_ic50.to(device)
        self.train_data = train_data.to(device)

        self.test_data = test_data.to(device)
        self.train_mask = train_mask.to(device)
        self.test_mask = test_mask.to(device)

        self.ind_ic50 = ind_ic50.to(device)
        self.ind_data = independ_data.to(device)
        self.ind_mask = ind_mask.to(device)

        self.evaluate_fun2 = evaluate_fun
        self.evaluate_scc = sc_score
        self.evaluate_rmse = rmse_score
        self.lr = lr
        self.wd = wd

        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )

    def forward(self):

        best_pcc = 0
        ind_pcc = 0
        ind_scc = 0
        ind_rmse = 0
        tole = 0
        pcc = 0

        true_data = torch.masked_select(self.test_ic50, self.test_mask)
        ind_true_data = torch.masked_select(self.ind_ic50, self.ind_mask)

        for epoch in torch.arange(self.epochs + 1):
            predict_data = self.model()
            loss = mse_loss(self.train_ic50, predict_data, self.train_mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            ind_data_masked = torch.masked_select(predict_data, self.ind_mask)
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            pcc = self.evaluate_fun2(true_data, predict_data_masked)

            if pcc > best_pcc:
                best_pcc = pcc
                best_predict = torch.masked_select(predict_data, self.ind_mask)
                ind_pcc = self.evaluate_fun2(ind_true_data, ind_data_masked)
                ind_scc = self.evaluate_scc(ind_true_data, ind_data_masked)
                ind_rmse = self.evaluate_rmse(ind_true_data, ind_data_masked)
                tole = 0
            else:
                tole += 1

            if epoch % self.test_freq == 0:
                print(
                    "epoch:%4d" % epoch.item(),
                    "loss:%.6f" % loss.item(),
                    "pcc:%.4f" % pcc,
                    "ind_pcc:%.4f" % ind_pcc,
                )

        print("Fit finished.")
        return ind_pcc, ind_scc, ind_rmse, ind_true_data, best_predict
