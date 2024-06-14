import torch
import numpy as np
import scipy.sparse as sp
from myutils import to_coo_matrix, to_tensor, mask


class RandomSampler_yz(object):
    # 对原始边进行采样
    # 采样后生成测试集、训练集
    # 处理完后的训练集转换为torch.tensor格式

    def __init__(self, adj_ic50, adj_binary, train_index, val_index, null_mask,test_index,test_neg_index):
        super(RandomSampler_yz, self).__init__()
        self.adj_mat = to_coo_matrix(adj_binary)
        self.adj_ic50 = torch.tensor(np.nan_to_num(adj_ic50))
        self.train_index = train_index
        self.val_index = val_index
        self.null_mask = null_mask
        
        self.test_neg_index = test_neg_index
        
        self.train_pos = self.sample(train_index)
        self.val_pos = self.sample(val_index)
        self.test_pos = self.sample(test_index)
        
        self.train_neg, self.val_neg, self.test_neg = self.sample_negative()
        self.train_mask = mask(self.train_pos, self.train_neg, dtype=int)
        self.val_mask = mask(self.val_pos, self.val_neg, dtype=bool)
        self.test_mask = mask(self.test_pos, self.test_neg, dtype=bool)
        
        self.train_data = to_tensor(self.train_pos)
        self.val_data = to_tensor(self.val_pos)
        self.test_data = to_tensor(self.test_pos)
    
        self.train_ic50 = torch.mul(self.train_mask, self.adj_ic50)
        self.val_ic50 = torch.mul(self.val_mask, self.adj_ic50)
        self.test_ic50 = torch.mul(self.test_mask, self.adj_ic50)
        
    def sample(self, index):
        row = self.adj_mat.row
        col = self.adj_mat.col
        data = self.adj_mat.data
        sample_row = row[index]
        sample_col = col[index]
        sample_data = data[index]
        sample = sp.coo_matrix((sample_data, (sample_row, sample_col)), shape=self.adj_mat.shape)
        return sample


    def sample_negative(self):
         
        # identity 表示邻接矩阵是否为二部图
        # 二部图：边的两个节点，是否属于同类结点集
        pos_adj_mat = self.null_mask + self.adj_mat.toarray()
        neg_adj_mat = sp.coo_matrix(np.abs(pos_adj_mat - np.array(1)))
        all_row = neg_adj_mat.row
        all_col = neg_adj_mat.col
        all_data = neg_adj_mat.data
        index = np.arange(all_data.shape[0])

# 采样独立负测试集
        test_row = all_row[self.test_neg_index]
        test_col = all_col[self.test_neg_index]
        test_data = all_data[self.test_neg_index]
        test = sp.coo_matrix((test_data, (test_row, test_col)), shape=self.adj_mat.shape)
        ind = np.zeros(all_data.shape[0], dtype=bool)
        ind[self.test_neg_index] = True
        index1 = index[~ind]
        # 采样负测试集
        val_n = self.val_index.shape[0]
        val_neg_index = np.random.choice(index1, val_n, replace=False)
        val_row = all_row[val_neg_index]
        val_col = all_col[val_neg_index]
        val_data = all_data[val_neg_index]
        val = sp.coo_matrix((val_data, (val_row, val_col)), shape=self.adj_mat.shape)
        ind[val_neg_index] = True
        index2 = index[~ind]
        # 采样训练集
        train_neg_index = index2
        train_row = all_row[train_neg_index]
        train_col = all_col[train_neg_index]
        train_data = all_data[train_neg_index]
        train = sp.coo_matrix((train_data, (train_row, train_col)), shape=self.adj_mat.shape)
        return train, val, test
    
class RandomSampler(object):
    # 对原始边进行采样
    # 采样后生成测试集、训练集
    # 处理完后的训练集转换为torch.tensor格式

    def __init__(self, adj_mat_original, train_index, test_index, null_mask):
        super(RandomSampler, self).__init__()
        self.adj_mat = to_coo_matrix(adj_mat_original)
        self.train_index = train_index
        self.test_index = test_index
        self.null_mask = null_mask
        self.train_pos = self.sample(train_index)
        self.test_pos = self.sample(test_index)
        self.train_neg, self.test_neg = self.sample_negative()
        self.train_mask = mask(self.train_pos, self.train_neg, dtype=int)
        self.test_mask = mask(self.test_pos, self.test_neg, dtype=bool)
        self.train_data = to_tensor(self.train_pos)
        self.test_data = to_tensor(self.test_pos)

    def sample(self, index):
        row = self.adj_mat.row
        col = self.adj_mat.col
        data = self.adj_mat.data
        sample_row = row[index]
        sample_col = col[index]
        sample_data = data[index]
        sample = sp.coo_matrix((sample_data, (sample_row, sample_col)), shape=self.adj_mat.shape)
        return sample

    def sample_negative(self):
        # identity 表示邻接矩阵是否为二部图
        # 二部图：边的两个节点，是否属于同类结点集
        pos_adj_mat = self.null_mask + self.adj_mat.toarray()
        neg_adj_mat = sp.coo_matrix(np.abs(pos_adj_mat - np.array(1)))
        all_row = neg_adj_mat.row
        all_col = neg_adj_mat.col
        all_data = neg_adj_mat.data
        index = np.arange(all_data.shape[0])

        # 采样负测试集
        test_n = self.test_index.shape[0]
        test_neg_index = np.random.choice(index, test_n, replace=False)
        test_row = all_row[test_neg_index]
        test_col = all_col[test_neg_index]
        test_data = all_data[test_neg_index]
        test = sp.coo_matrix((test_data, (test_row, test_col)), shape=self.adj_mat.shape)

        # 采样训练集
        train_neg_index = np.delete(index, test_neg_index)
        train_row = all_row[train_neg_index]
        train_col = all_col[train_neg_index]
        train_data = all_data[train_neg_index]
        train = sp.coo_matrix((train_data, (train_row, train_col)), shape=self.adj_mat.shape)
        return train, test
