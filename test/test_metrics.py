import torch
from RecSys.metrics import HITS_K, MRR_K, Accuracy_K, Recall_K, NDCG_K, HITS_K_first_bootstrap
import math


def test_Accuracy_k_e():
    metric = Accuracy_K(4, exclude=True)

    # test 1
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = torch.tensor([[True, False, False, False, True]])
    exclude = torch.tensor([[False, False, False, True, False]])
    metric.update(preds, target, exclude)
    assert metric.compute() == 0.5
    exclude = torch.tensor([[False, False, False, False, False]])
    metric.update(preds, target, exclude)
    assert metric.compute() == 0.25

    # test 2
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                          [0.8, 0.7, 0.6, 0.5, 0.4]])
    target = torch.tensor([[False, True, False, True, False],
                           [False, True, False, False, True]])
    exclude = torch.tensor([[False, False, False, True, False],
                            [False, False, True, False, False]])
    metric.update(preds, target, exclude)
    assert metric.compute() == 3 / 8


def test_Accuracy_k_ne():
    metric = Accuracy_K(4, exclude=False)

    # test 1
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = torch.tensor([[False, True, False, False, True]])
    metric.update(preds, target, None)

    assert metric.compute() == 0.5

    # test 2
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                          [0.8, 0.7, 0.6, 0.5, 0.4]])
    target = torch.tensor([[False, True, False, True, False],
                           [False, True, False, False, True]])
    metric.update(preds, target, None)
    assert metric.compute() == 3 / 8


def test_Recall_k_ne():
    metric = Recall_K(4, exclude=False)

    # test 1
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = torch.tensor([[False, True, False, False, True]])
    metric.update(preds, target, None)

    assert metric.compute() == 1

    # test 2
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = torch.tensor([[True, True, False, False, True]])
    metric.update(preds, target, None)

    assert metric.compute() == 2 / 3

    # test 3
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                          [0.9, 0.8, 0.7, 0.6, 0.5]])
    target = torch.tensor([[False, True, False, False, False],
                           [False, True, False, False, True]])
    metric.update(preds, target, None)
    assert metric.compute() == 0.5 * (1 + 1/2)

    # test 4
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = torch.tensor([[False, False, False, False, False]])
    metric.update(preds, target, None)
    print(metric.compute())


def test_Hits_k_ne():
    metric = HITS_K(4, exclude=False)

    # test 1
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = torch.tensor([[False, True, False, False, True]])
    metric.update(preds, target, None)

    assert metric.compute() == 1

    # test 2
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                          [0.6, .7, 0.8, 0.9, 1]])
    target = torch.tensor([[False, True, False, False, False],
                           [True, False, False, False, False]])
    metric.update(preds, target, None)
    assert metric.compute() == 1 / 2


def test_mrr_k_ne():
    metric = MRR_K(4, exclude=False)

    # test 1
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = torch.tensor([[False, True, False, False, True]])
    metric.update(preds, target, None)

    assert metric.compute() == (1 / 4 + 1) / 2

    # test 2
    preds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, .7, 0.8, 0.9, 1])
    preds = preds.reshape(2, 5)
    assert (preds == torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, .7, 0.8, 0.9, 1]])).all()
    target = torch.tensor([[False, True, False, False, False], [False, True, False, False, True]])
    metric.update(preds, target, None)
    assert metric.compute() == ((1/4) + ((1 + (1/4))/2)) / 2


def test_ndcg_k_ne():
    metric = NDCG_K(4, exclude=False)

    # test 1
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = torch.tensor([[False, True, False, False, True]])
    metric.update(preds, target, None)

    assert metric.compute() == (1/math.log2(2) + 1/math.log2(5)) / (1/math.log2(2) + 1/math.log2(3))

    # test 2
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = torch.tensor([[False, False, False, False, False]])
    metric.update(preds, target, None)

    assert metric.compute() == 0

    # test 3
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = torch.tensor([[False, False, False, True, True]])
    metric.update(preds, target, None)

    assert metric.compute() == 1

    # test 4
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                          [0.9, 0.8, 0.7, 0.6, 0.5]])
    target = torch.tensor([[True, False, False, False, True],
                           [False, True, False, False, False]])
    metric.update(preds, target, None)

    m1 = (1/math.log2(2)) / (1/math.log2(2) + 1/math.log2(3))
    m2 = (1/math.log2(3)) / (1/math.log2(2))
    assert metric.compute() == (m1 + m2) / 2


def test_hits_k_ne():
    metric = HITS_K(4, exclude=False)

    # test 1
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = torch.tensor([[False, True, False, False, True]])
    metric.update(preds, target, None)

    assert metric.compute() == 1

    # test 2
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = torch.tensor([[True, False, False, False, False]])
    metric.update(preds, target, None)

    assert metric.compute() == 0

    # test 3
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.3]])
    target = torch.tensor([[False, False, False, False, True]])
    metric.update(preds, target, None)

    assert metric.compute() == 1

    # test 4
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                          [0.9, 0.8, 0.7, 0.6, 0.5]])
    target = torch.tensor([[True, False, False, False, True],
                           [False, True, False, False, False]])
    metric.update(preds, target, None)

    assert metric.compute() == 1

    # test 5
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                          [0.9, 0.8, 0.7, 0.6, 0.5]])
    target = torch.tensor([[True, False, False, False, True],
                           [False, False, False, False, True]])
    metric.update(preds, target, None)

    assert metric.compute() == 0.5


def test_hits_k_paper():
    metric = HITS_K_first_bootstrap(3, num_rep=100)

    # test 1
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = None
    exclude = torch.tensor([[False, False, False, True, False]])
    first_inter = torch.tensor([[2]])

    metric.update(preds, target, exclude, first_inter)
    assert metric.compute() >= 0.99

    # test 2
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    target = None
    exclude = torch.tensor([[False, False, False, True, False]])
    first_inter = torch.tensor([[0]])

    metric.update(preds, target, exclude, first_inter)
    assert metric.compute() <= 0.01

    # test 3
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                          [0.9, 0.8, 0.7, 0.6, 0.5]])
    target = None
    exclude = torch.tensor([[False, False, False, True, False],
                            [False, False, False, True, False]])
    first_inter = torch.tensor([[0], [2]])

    metric.update(preds, target, exclude, first_inter)
    assert 0.49 <= metric.compute() <= 0.51
