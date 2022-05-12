import torch
from scipy import sparse


def nDCG(ground_truth,
        scores = None,
        ufeats = None,
        ifeats = None,
        k = 100,
        batch_size = 100,
        end = 10000,
        train = None,
        item_keys = None
        ):
    """
    Parameters
    ----------
    ground_truth : scipy.sparse.csr_matrix
        (n_users, n_items)
    scores : torch.FloatTensor
        predicted relevance
        it can be replaced by ufeats & ifeats
    ufeats : torch.FloatTensor()
        (n_users, hidden_dim)
    ifeats : torch.FloatTensor()
        (hidden_dim, n_items)
    train : scipy.sparse.csr_matrix
        (n_users, n_items)
        Optional

    Examples
    --------
    decay = [1.0000, 0.6309, 0.5000, 0.4307, 0.3869, ....] where decay[i] = 1/log2(i + 2)

    1. binary relevance
        - scores       = [0, 3, 0, 5, 9, 0, 0] => relevance : [0, 1, 0, 1, 1, 0, 0]
                        => top3 : [4,3,1]
        - ground_truth = [9, 3, 0, 0, 5, 0, 0] => relevance : [1, 1, 0, 0, 1, 0, 0]
        
            -> dcg = 1 * decay[0] + 0 * decay[1] + 1 * decay[2] = 1.5000
            -> ideal_dcg = 1 * decay[0] + 1 * decay[1] + 1 * decay[2] = 2.1309

        return ndcg = 1.5 / 2.1309
    
    2. non binary relevance
        - scores       = [0, 3, 0, 5, 9, 0, 0]
                        => top3 : [4,3,1]
        - ground_truth = [9, 3, 0, 0, 5, 0, 0]
        
            -> dcg = 5 * decay[0] + 0 * decay[1] + 3 * decay[2] = 6.5
            -> ideal_dcg = 9 * decay[0] + 5 * decay[1] + 3 * decay[2] = 13.6545

        return ndcg = 6.5 / 13.6545

    """
    assert scores is not None or (ufeats is not None and ifeats is not None)

    if scores is not None:
        if item_keys is not None:
            scores = scores[:, item_keys]
        return nDCG_per_user(ground_truth, scores, k).mean().item()

    if item_keys is not None:
        ifeats = ifeats[item_keys]
    ifeats_t = ifeats.t()

    ndcgs = []
    for start in range(0, end - batch_size, batch_size):
        scores = ufeats[start:start + batch_size] @ ifeats_t

        gt_batch = ground_truth[start:start + batch_size]

        train_batch = train[start:start + batch_size]
        scores[train_batch.nonzero()] = float('-inf')

        ndcgs.append(nDCG_per_user(gt_batch, scores, k = k))

    ndcg = torch.cat(ndcgs).mean().item()

    return ndcg

def nDCG_per_user(ground_truth, scores, k = 100):
    """
    Parameters
    ----------
    scores : torch.FloatTensor
        predicted relevance
        (batch_size, n_items)
    ground_truth : scipy.sparse.csr_matrix
        (batch_size, n_items)
    """

    batch_size = scores.shape[0]

    # dcg
    _, pred_rank = torch.topk(torch.Tensor(scores), k = k, axis = 1)
    decay = torch.pow(torch.arange(2,k + 2).log2(), -1)
    dcg = ground_truth[torch.arange(0,batch_size).unsqueeze(1), pred_rank] @ decay

    # idcg
    gt = torch.Tensor(ground_truth.todense())
    gt_relevance, gt_rank = torch.topk(gt, k = k, axis = 1)
    mask = gt_relevance >= 0

    idcg = gt[torch.arange(0,batch_size).unsqueeze(1), gt_rank] * mask
    idcg = idcg @ decay

    ndcg = dcg[idcg > 0.]/idcg[idcg > 0.]

    return ndcg

def short_head(train,
            scores = None,
            ufeats = None,
            ifeats = None,
            ratio = 0.2,
            k = 100,
            batch_size = 100,
            end = 10000,
            rm_seen_item = True
            ):
    """
    Parameters
    ----------
    train : scipy.sparse.csr_matrix
        (n_users, n_items)
    scores : torch.FloatTensor
        predicted score
        it can be replaced by ufeats & ifeats
    ufeats : torch.FloatTensor()
        (n_users, hidden_dim)
    ifeats : torch.FloatTensor()
        (hidden_dim, n_items)

    Examples
    --------
    train = [[1 0 1 1 0 0 1]
             [0 0 0 1 0 1 1]]
        => clicks_per_item = [1, 0, 1, 2, 0, 2]
        => most popular items @ 2 = [3, 6]

    scores = [0, 3, 0, 5, 9, 0, 0]
        => top @ 3 : [4, 3, 1]

    return short_head_ratio = 1 / 3
    """
    assert scores is not None or (ufeats is not None and ifeats is not None)

    n_items = train.shape[1]
    short_head_size = int(n_items * ratio)

    n_clicks = torch.tensor(train.sum(axis = 0)).flatten()
    most_popular_items = torch.topk(n_clicks, k = short_head_size)[1]
    non_most_popular = torch.BoolTensor([1] * n_items)
    non_most_popular[most_popular_items] = False
    
    if scores is not None:
        return short_head_per_user(scores, non_most_popular, k = k).mean().item()

    short_head_ratio = []
    ifeats_t = ifeats.t()

    for start in range(0, end - batch_size , batch_size):
        scores = ufeats[start:start + batch_size] @ ifeats_t

        if rm_seen_item:
            train_batch = train[start:start + batch_size]
            scores[train_batch.nonzero()] = float('-inf')

        short_head_ratio.append(short_head_per_user(scores, most_popular_items, k))

    return torch.cat(short_head_ratio).mean().item()

def short_head_per_user(scores, non_most_popular, k = 100):
    """
    Parameters
    ----------
    scores : torch.FloatTensor
        predicted relevance
        (batch_size, n_items)
    non_most_popular : torch.BoolTensor
        (n_items, )
    """
    _, pred_rank = torch.topk(scores, k = k)

    short_head = torch.zeros_like(scores)
    short_head[torch.arange(pred_rank.shape[0]).unsqueeze(1), pred_rank] = 1

    short_head.masked_fill_(non_most_popular, 0.)

    return short_head.sum(axis = -1) / k

if __name__ == '__main__':
    train = torch.tensor([
        [1, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 1]
        ])
    train = sparse.csr_matrix(train)

    test = torch.tensor([
        [9, 3, 0, 0, 5, 0, 0],
        [1, 1, 0, 0, 1, 0, 0]
        ])
    test = sparse.csr_matrix(test)

    scores = torch.FloatTensor([
        [0, 3, 0, 5, 9, 0, 0],
        [0, 3, 0, 5, 9, 0, 0]
        ])
    
    ndcg = nDCG(ground_truth = test, scores = scores, k = 3)
    print('ndcg : ', ndcg) # mean([0.4760, 0.7039])

    sh = short_head(train, scores, k = 3)
    print('short head ratio : ', sh)
