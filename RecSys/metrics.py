""" Metrics for recommendation tasks. """
from abc import ABC, abstractmethod
from typing import List, Union
import torch
import torchmetrics
from torch import Tensor
from tqdm import tqdm


class LossMetric(torchmetrics.Metric):
    """ LossMetric : compute the loss on a batch """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Loss"
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss, preds_pos, preds_neg):
        self.loss += loss * len(preds_pos)
        self.n += len(preds_pos)  # type: ignore

    def compute(self) -> Tensor:
        return self.loss / self.n  # type: ignore


class RankingAccuracy(torchmetrics.Metric):
    """ RankingAccuracy : compute proportion of positive predictions that are higher than negative predictions """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Ranking Accuracy"
        self.add_state("hits", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds_pos, preds_neg):
        self.hits += (preds_pos > preds_neg).sum()
        self.n += preds_pos.shape[0]

    def compute(self) -> Tensor:
        return self.hits / self.n  # type: ignore


class AUCROC(torchmetrics.Metric):
    """ AUCROC : compute the area under the ROC curve for the positive and negative predictions """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "ROC"
        self.add_state("auc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds_pos, preds_neg):
        preds_pos = torch.sigmoid(preds_pos)
        preds_neg = torch.sigmoid(preds_neg)
        preds = torch.cat([preds_pos, preds_neg], dim=0)
        target = torch.cat([torch.ones_like(preds_pos, dtype=torch.int32), torch.zeros_like(preds_neg, dtype=torch.int32)], dim=0)
        self.auc += torchmetrics.functional.auroc(preds=preds, target=target, pos_label=1)  # type: ignore
        self.n += 1  # type: ignore

    def compute(self) -> Tensor:
        return self.auc / self.n  # type: ignore


class TestMetric(ABC):
    """ TestMetric : compute the metrics on the test set """
    name, metric = "", Tensor(0)

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, preds: Tensor, target: Tensor, exclude: Union[Tensor, None], *args, **kwargs) -> None:
        raise NotImplementedError

    def compute(self) -> Tensor:
        return torch.mean(self.metric)


class Accuracy_K(TestMetric):
    """Accuracy@K metric."""

    def __init__(self, k, exclude=True, *args, **kwargs):
        self.name = "Acc@" + str(k)
        self.name += "_e" if exclude else "_ne"
        self.k = k
        self.exclude = exclude

    def update(self, preds, target, exclude, *args, **kwargs):
        if self.exclude:
            assert exclude is not None
            preds = torch.where(exclude, preds.min()-1, preds)
        hits = torch.sum(target.take_along_dim(preds.topk(k=self.k, dim=-1)[1], dim=-1), dim=-1)
        n = self.k
        self.metric = hits / n
        self.metric.cpu()


class Recall_K(TestMetric):
    """Recall@K metric."""

    def __init__(self, k, exclude=True, *args, **kwargs):
        self.name = "Rec@" + str(k)
        self.name += "_e" if exclude else "_ne"
        self.k = k
        self.exclude = exclude

    def update(self, preds, target, exclude, *args, **kwargs):
        if self.exclude:
            assert exclude is not None
            preds = torch.where(exclude, preds.min()-1, preds)
        hits = torch.sum(target.take_along_dim(preds.topk(k=self.k, dim=-1)[1], dim=-1), dim=-1)
        hits = hits.nan_to_num_(nan=0.0)
        n = torch.sum(target, dim=-1)
        n = torch.where(n == 0, torch.ones_like(n), n)
        self.metric = hits / n
        self.metric.cpu()


class MRR_K(TestMetric):
    """Mean reciprocal rank metric."""

    def __init__(self, k, exclude=True, *args, **kwargs):
        self.name = "MRR@" + str(k)
        self.name += "_e" if exclude else "_ne"
        self.k = k
        self.exclude = exclude

    def update(self, preds, target, exclude, *args, **kwargs):
        if self.exclude:
            assert exclude is not None
            preds = torch.where(exclude, preds.min()-1, preds)
        ranks = target.take_along_dim(preds.topk(k=self.k, dim=-1)[1], dim=-1)
        ranks = torch.where(ranks)[-1] + 1
        # self.hits = torch.sum(1/ranks)
        hits = 1/ranks
        n = torch.sum(target, dim=-1)
        n = torch.where(n > self.k, self.k, n)
        # self.n = torch.sum(self.n)
        self.metric = hits / n
        self.metric.cpu()


class NDCG_K(TestMetric):
    """Mean reciprocal rank metric."""

    def __init__(self, k, exclude=True, *args, **kwargs):
        self.name = "NDCG@" + str(k)
        self.name += "_e" if exclude else "_ne"
        self.k = k
        self.exclude = exclude

    def update(self, preds, target, exclude, *args, **kwargs):
        if self.exclude:
            assert exclude is not None
            preds = torch.where(exclude, preds.min()-1, preds)

        mask_ranks = target.take_along_dim(preds.topk(k=self.k, dim=-1)[1], dim=-1)
        ranks = torch.arange(1, self.k+1, device=mask_ranks.device)
        dcg = 1 / torch.log2(ranks + 1)
        dcg = torch.where(mask_ranks, dcg, torch.zeros_like(dcg))
        dcg = torch.sum(dcg, dim=-1)

        n = torch.sum(target, dim=-1)
        n = torch.where(n > self.k, self.k, n)

        ideal_ranks = torch.arange(1, self.k+1, device=ranks.device)
        all_idcg = [torch.sum(1 / torch.log2(ideal_ranks[:n] + 1)) for n in range(1, self.k+1)]
        all_idcg = [torch.tensor(1.0, device=ranks.device)] + all_idcg
        idcg = []
        for n in n.cpu().numpy():
            idcg.append(all_idcg[n])
        idcg = torch.tensor(idcg, device=ranks.device)
        self.metric = dcg / idcg
        self.metric.cpu()


class HITS_K(TestMetric):
    """Hits@K metric."""

    def __init__(self, k, exclude=True, *args, **kwargs):
        self.name = "HITS@" + str(k)
        self.name += "_e" if exclude else "_ne"
        self.k = k
        self.exclude = exclude

    def update(self, preds, target, exclude, *args, **kwargs):
        if self.exclude:
            assert exclude is not None
            preds = torch.where(exclude, preds.min()-1, preds)
        hits = torch.sum(target.take_along_dim(preds.topk(dim=-1, k=self.k)[1], dim=-1), dim=-1)
        hits = torch.where(hits > 0, torch.ones_like(hits), hits)
        hits = hits.to(dtype=torch.float32)
        self.metric = hits
        self.metric.cpu()


class HITS_K_first_bootstrap(TestMetric):
    """ HITS@K metric as in the conference paper. """

    def __init__(self, k, num_rep=50) -> None:
        self.name = "HITS@" + str(k) + "_first_bootstrap"
        self.k = k
        self.num_rep = num_rep

    def update(self, preds, target, exclude, first_inter, *args, **kwargs):
        hits = torch.zeros((preds.size(0),), dtype=torch.float32, device=preds.device)
        self.metric = torch.zeros((preds.size(0),), dtype=torch.float32, device=preds.device)
        probs = torch.where(exclude, torch.zeros_like(exclude, dtype=torch.float32), torch.ones_like(exclude, dtype=torch.float32))
        for _ in tqdm(range(self.num_rep)):
            # Sample 100 items users have not interacted with
            sampled_items = torch.multinomial(probs, 99, replacement=True)

            # compute the k/100 quantile of the scores of the sampled items
            sampled_scores = preds.take_along_dim(sampled_items, dim=-1)
            quantile = torch.quantile(sampled_scores, self.k/100, dim=-1)

            # compute the hits@k
            score_first_item = preds.take_along_dim(first_inter, dim=-1).squeeze(-1)
            hits = torch.where(score_first_item > quantile, torch.ones_like(score_first_item), torch.zeros_like(score_first_item))
            hits = hits.to(dtype=torch.float32)
            self.metric += hits / self.num_rep
        self.metric.cpu()


class NDCG_K_first_bootstrap(TestMetric):
    """ NDCG@K metric as in the conference paper. """

    def __init__(self, k, num_rep=50) -> None:
        self.name = "NDCG@" + str(k) + "_first_bootstrap"
        self.k = k
        self.num_rep = num_rep

    def update(self, preds, target, exclude, first_inter, *args, **kwargs):
        dcg = torch.zeros((preds.size(0),), dtype=torch.float32, device=preds.device)
        probs = torch.where(exclude, torch.zeros_like(exclude, dtype=torch.float32), torch.ones_like(exclude, dtype=torch.float32))
        for _ in tqdm(range(self.num_rep)):
            # Sample 100 items users have not interacted with
            sampled_items = torch.multinomial(probs, 99, replacement=True)

            # compute the k/100 quantile of the scores of the sampled items
            sampled_scores = preds.take_along_dim(sampled_items, dim=-1)

            # compute the rank
            score_first_item = preds.take_along_dim(first_inter, dim=-1)
            sampled_scores = torch.cat((sampled_scores, score_first_item), -1)
            ranks = torch.argsort(sampled_scores, dim=-1, descending=True)
            ranks = torch.where(ranks == 99)[1]
            dcg += 1 / torch.log2(ranks + 2)
        dcg /= self.num_rep
        self.metric = dcg
        self.metric.cpu()


class HITS_K_first(TestMetric):
    """Hits@K metric of the first element the users have interacted with in the test dataset."""

    def __init__(self, k, *args, **kwargs):
        self.name = "HITS@" + str(k) + "_first"
        self.k = k

    def update(self, preds, target, exclude, first_inter, *args, **kwargs):
        topk = preds.topk(dim=-1, k=self.k)[1]
        hits = [torch.isin(first_inter[i][0], topk[i]) for i in range(preds.size(0))]
        hits = torch.tensor(hits, dtype=torch.float32, device=preds.device)
        self.metric = hits
        self.metric.cpu()


class RecommendationExample():
    """ Show the top recommendation of a user """

    def __init__(self, user=0, nb_preds=10):
        self.name = "Recommendation Example for user " + str(user)
        self.user = user
        self.nb_preds = nb_preds

    def update(self, preds, target, exclude, first_inter, *args, **kwargs):
        preds = preds[self.user]
        topk = preds.topk(dim=0, k=self.nb_preds)
        self.topk = topk

    def compute(self):
        return self.topk


class MetricsPrint():
    """ Class to visualize metrics during training """

    def __init__(self, metrics: List[torchmetrics.Metric]):
        self.metrics_names = [m.name for m in metrics]

    def initial_print(self, nbatch, name=" "*16):
        """Initial print on the beginig of an epoch"""
        self.nbatch = nbatch
        text = name
        for name in self.metrics_names:
            n = 12 - len(name)  # type: ignore
            text += "|" + " " * (n//2) + name + " " * (n//2 + n % 2)  # type: ignore
        print(text)

    def running_print(self, metrics_values, batch):
        """Print during an epoch"""
        if batch % int(self.nbatch / 10) == 0:
            text = f"{batch:>6} / {self.nbatch:>6} "
            for v in metrics_values:
                text += f"|  {v:.6f}  "
            print(text)

    def final_print(self, metrics_values):
        self.initial_print(self.nbatch)
        text = "   Finally:     "
        for v in metrics_values:
            text += f"|  {v:.6f}  "
        print(text, "\n")
