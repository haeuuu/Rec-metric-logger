import numpy as np

from metric import *


class MetricLogger:
    def __init__(self,
                train,
                test,
                k_for_ndcg = 100,
                k_for_sh = 100,
                short_head_ratio = 0.1,
                end = 10000
                ):
        """Metric logger for recommendation systems

        Paramters
        ---------
        train, test : scipy.sparse.csr_matrix
            (n_users, n_items)
        short_head_ratio : float
            short head ratio
        k_for_ndcg : int
            calculate nDCG @ k_for_ndcg
        k_for_sh : int
            calculate ShortHead @ k_for_sh
        """
        self.train = train
        self.test = test

        self.short_head_ratio = short_head_ratio
        self.k_for_ndcg = k_for_ndcg
        self.k_for_sh = k_for_sh
        self.end = end if end > 0 else train.shape[0]

        self.truncate()

    def truncate(self):
        """truncate train/test data to calculate nDCG using only non-short head items
        """
        n_clicks = torch.tensor(self.train.sum(axis = 0)).flatten()
        short_head_items = int(self.train.shape[1] * self.short_head_ratio)
        _, topk_idx = torch.topk(n_clicks, short_head_items)
        self.item_truncated = ~np.isin(range(self.train.shape[1]), topk_idx)

        self.train_truncated = self.train[:,self.item_truncated]
        self.test_truncated = self.test[:,self.item_truncated]

    def summary(self, scores = None, ufeats = None, ifeats = None):
        """
        Parameters
        ----------
        scores : torch.FloatTensor
            predicted relevance
            it can be replaced by ufeats & ifeats
        ufeats : torch.FloatTensor()
            (n_users, hidden_dim)
        ifeats : torch.FloatTensor()
            (hidden_dim, n_items)
        """
        ndcg = nDCG(ground_truth = self.test,
                    scores = scores,
                    ufeats = ufeats,
                    ifeats = ifeats,
                    k = self.k_for_ndcg,
                    end = self.end,
                    train = self.train)

        ndcg_tc = nDCG(ground_truth = self.test_truncated,
                    scores = scores,
                    ufeats = ufeats,
                    ifeats = ifeats,
                    k = self.k_for_ndcg,
                    end = self.end,
                    train = self.train_truncated,
                    item_keys = self.item_truncated)

        sh = short_head(train = self.train,
                        scores = scores,
                        ufeats = ufeats,
                        ifeats = ifeats,
                        ratio = self.short_head_ratio,
                        k = self.k_for_sh,
                        end = self.end,
                        rm_seen_item = True)

        return ndcg, ndcg_tc, sh

    def logging(self, scores = None, ufeats = None, ifeats = None):
        """
        Parameters
        ----------
        scores : torch.FloatTensor
            predicted relevance
            it can be replaced by ufeats & ifeats
        ufeats : torch.FloatTensor()
            (n_users, hidden_dim)
        ifeats : torch.FloatTensor()
            (hidden_dim, n_items)
        """
        ndcg, ndcg_tc, short_head = self.summary(scores, ufeats, ifeats)

        log = \
        f"nDCG @ {self.k_for_ndcg} / Short Head @ {self.k_for_sh}\n" + \
        f"validation nDCG                   : {ndcg:.4f}\n" + \
        f"validation nDCG (truncated)       : {ndcg_tc:.4f}\n" + \
        f"Short Head ratio                  : {short_head:.4f}\n"

        return log

if __name__ == '__main__':
    from scipy import sparse

    train = torch.tensor([
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 1]
        ])
    train = sparse.csr_matrix(train)

    test = torch.tensor([
        [1, 0, 0, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0]
        ])
    test = sparse.csr_matrix(test)

    scores = torch.tensor([
        [0, 1, 9, 3, 0, 2, 3],
        [0, 6, 5, 5, 0, 1, 1],
        [1, 0, 0, 9, 0, 3, 2]
        ]) * 0.1

    logger = MetricLogger(train, test, k_for_ndcg = 3, k_for_sh = 3, short_head_ratio = 0.3, end = -1)
    log = logger.logging(scores = scores)
    
    print(log)