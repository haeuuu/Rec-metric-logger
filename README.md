# 📝 Rec-Metric-logger
**Metric logger for recommendation systems**
  
</br>

## **Examples**
```python
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

logger = MetricLogger(train,
                        test,
                        k_for_ndcg = 3,
                        k_for_sh = 3,
                        short_head_ratio = 0.3,
                        end = -1)

log = logger.logging(scores = scores)

print(log)
```
```
nDCG @ 3 / Short Head @ 3
validation nDCG                   : 0.4623
validation nDCG (truncated)       : 0.5617
Short Head ratio                  : 0.5556
```
  
</br>

## `metric.py`
- [v] nDCG
- [v] short head ratio
- [ ] precision
- [ ] recall
- [ ] confusion matrix

#### **`nDCG @ k`**
```python
from metric import nDCG

nDCG(ground_truth = ground_truth,
    scores = predicted_relevance,
    k = k)

nDCG(ground_truth = ground_truth,
    ufeats = user_features,
    ifeats = item_features,
    k = k) # it will use ufeats @ ifeats.t() as predicted_relevance
```
```
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
```

#### **`Short Head @ k`**
```python
from metric import short_head

short_head(train = train,
            scores = predicted_relevance,
            ratio = ratio,
            k = k)

short_head(train = train,
            ufeats = user_features,
            ifeats = item_features,
            ratio = ratio,
            k = k) # it will use ufeats @ ifeats.t() as predicted_relevance
```
```
train = [[1 0 1 1 0 0 1]
        [0 0 0 1 0 1 1]]
    => clicks_per_item = [1, 0, 1, 2, 0, 2]
    => most popular items @ 2 = [3, 6]

scores = [0, 3, 0, 5, 9, 0, 0]
    => top @ 3 : [4, 3, 1]

return short_head_ratio = 1 / 3
```