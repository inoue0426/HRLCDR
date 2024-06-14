from models import hrlcdr_new, Optimizer_new
from myutils import *



def HRLCDR_new( cell_exprs, drug_finger, sam,test_da,test_ma,
                args,cell_hyper,drug_hyper,A1,A2,A3,B1,B2,B3,
                        device,alp,bet):

        model = hrlcdr_new(adj_mat=sam.train_data,  cell_exprs=cell_exprs, drug_finger=drug_finger,
                gamma=args.gamma, drug_hyper=drug_hyper,cell_hyper=cell_hyper, device=args.device,alp=alp,bet=bet).to(args.device)

        opt = Optimizer_new(model,  sam.train_data, sam.test_data, sam.test_mask,sam.train_mask,
                        test_da, test_ma,roc_auc,ap_score,
                        lr=args.lr, wd=args.wd, epochs=args.epochs,tol=args.tol, device=args.device).to(args.device)

        best_auc,best_aupr, true_data,predict_data= opt()

        return best_auc,best_aupr,true_data, predict_data
