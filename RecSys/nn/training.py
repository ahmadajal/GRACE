""" Training utils and losses """
from abc import ABC, abstractmethod
from typing import List, Union, Tuple
from time import time

import torch
import torchmetrics
from torch import Tensor

from tqdm import tqdm

from RecSys.metrics import MetricsPrint
from RecSys.utils.data.data import RecGraphData
from RecSys.nn.models import GNN
from RecSys.nn.models import CF


class Loss(ABC):
    """ Loss function """

    @abstractmethod
    def __call__(self, preds_pos: Tensor, preds_neg: Tensor, weight_norm: Union[Tensor, None]) -> Tensor:
        raise NotImplementedError

    def __call_on_batch_gnn(self, model: GNN, graph: RecGraphData, batch: Tuple[Tensor, Tensor, Tensor]):
        """ Compute the loss on a batch for a GNN model
        Args:
            model (GNN): GNN model.
            batch (Tuple): Batch.
        Returns:
            Tensor: BPR loss.
        """
        out = model.get_embedding(graph)
        i, j, k = batch[0].to(out.device), batch[1].to(out.device), batch[2].to(out.device)
        preds_pos = (out[i] * out[j]).sum(dim=1)
        preds_neg = (out[i] * out[k]).sum(dim=1)
        # return self(preds_pos, preds_neg, model.embedding.weight.norm(p=2)), (preds_pos, preds_neg)
        return self(preds_pos, preds_neg, model.get_weight_norm()), (preds_pos, preds_neg)

    def __call_on_batch_cf(self, model: CF, graph: RecGraphData, batch):
        """ Compute the loss on a batch for a CF model
        Args:
            model (CF): CF model.
            batch (Data): Batch.
        Returns:
            Tensor: BPR loss.
        """
        i, j, k = batch[0].to(graph.x.device), batch[1].to(graph.x.device), batch[2].to(graph.x.device)
        preds_pos = model(graph, i, j)
        if len(k.shape) == 2:
            preds_neg = torch.cat([model(graph, i, k[:, col]) for col in range(k.size(1))], 0)
        else:
            preds_neg = model(graph, i, k)
        return self(preds_pos, preds_neg, model.get_weight_norm()), (preds_pos, preds_neg)

    def call_on_batch(self, model: Union[GNN, CF], graph: RecGraphData, batch):
        """ Compute the loss on a batch
        Args:
            model (Union[GNN, CF]): Model.
            batch (Data): Batch.
        Returns:
            Tensor: BPR loss.
        """
        if isinstance(model, GNN):
            return self.__call_on_batch_gnn(model, graph, batch)
        if isinstance(model, CF):
            return self.__call_on_batch_cf(model, graph, batch)
        else:
            raise ValueError("Model must be either a subclass of GNN or CF")


class BPRLoss(Loss):
    """ BPRLoss """

    def __init__(self, weight_decay: float = 0.0):
        """ BPRLoss constructor
        Args:
            weight_decay (float): Weight decay.
        """
        self.weight_decay = weight_decay

    def __call__(self, preds_pos: Tensor, preds_neg: Tensor, weight_norm: Union[Tensor, None] = None) -> Tensor:
        """ Compute the BPR loss
        Args:
            preds_pos (Tensor): Predictions for positive samples.
            preds_neg (Tensor): Predictions for negative samples.
            weight_norm (Tensor): Weight norm.
        Returns:
            Tensor: BPR loss.
        """
        if weight_norm is not None:
            reg = self.weight_decay * weight_norm
        else:
            reg = 0
        return - torch.nn.functional.logsigmoid(preds_pos - preds_neg).mean() + reg


class CELoss(Loss):
    """ CELoss """

    def __init__(self, weight_decay: float = 0.0):
        """ CELoss constructor
        Args:
            weight_decay (float): Weight decay.
        """
        self.weight_decay = weight_decay

    def __call__(self, preds_pos: Tensor, preds_neg: Tensor, weight_norm: Union[Tensor, None] = None) -> Tensor:
        """ Compute the CE loss
        Args:
            preds_pos (Tensor): Predictions for positive samples.
            preds_neg (Tensor): Predictions for negative samples.
            weight_norm (Tensor): Weight norm.
        Returns:
            Tensor: CE loss.
        """
        if weight_norm is not None:
            reg = self.weight_decay * weight_norm
        else:
            reg = 0
        preds_neg = torch.flatten(preds_neg)
        preds = torch.cat((preds_pos, preds_neg), 0)
        target = torch.cat((torch.ones_like(preds_pos), torch.zeros_like(preds_neg)), 0)
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, target)
        return ce_loss + reg


def train_fn(model: Union[GNN, CF], graph: RecGraphData,
             optimizer, scheduler, epoch_per_lr_decay, dataloader, loss_fn, epoch_begin: int = 1,
             nepochs=1, metrics: Union[list, None] = None, exp=None):
    """ Train a model
    Args:
        model (GNN | CF): Model to train.
        graph (RecGraphData): graph we train on.
        optimizer (Optimizer): Optimizer.
        scheduler (Scheduler): Scheduler.
        epoch_per_lr_decay (int): Number of epochs per lr decay.
        dataloader (DataLoader): Train dataloader.
        loss_fn (Callable): Loss function.
        epoch_begin (int): Epoch to start from.
        nepochs (int): Number of epochs.
        metrics (List[torchmetrics.Metric]): Metrics to print.
        exp (Experiment | None): Experiment to save results.
    """
    metrics = metrics or []
    # Train mode
    model.train()

    # Metrics print
    metrics_printer = MetricsPrint(metrics)

    # Loop over epochs
    for epoch in range(epoch_begin, epoch_begin+nepochs):
        metrics_printer.initial_print(len(dataloader), name=f"Epoch {epoch}" + (9-epoch//10)*" ")

        # Reset metrics
        for metric in metrics:
            metric.reset()

        # Sample new negative items
        dataloader.dataset.compute_neg_items()

        # Iterate over batches
        for batch_idx, batch in enumerate(dataloader):
            # Training routine
            optimizer.zero_grad()
            loss, preds = loss_fn.call_on_batch(model, graph, batch)
            loss.backward()
            optimizer.step()

            # Update metrics
            for met in metrics:
                met.update(loss, *preds)
            metrics_printer.running_print([metric.compute() for metric in metrics], batch_idx)

        # Print metrics
        metrics_printer.final_print([m.compute() for m in metrics])
        if exp:
            exp.on_epoch_end(epoch, metrics)
            exp.save()

        # Update scheduler
        if scheduler is not None:
            if epoch % epoch_per_lr_decay == 0:
                scheduler.step()


@torch.no_grad()
def __test_fn_gnn(model: GNN, graph: RecGraphData,
                  test_ds, epoch: int,
                  metrics: Union[List[torchmetrics.Metric], None] = None,
                  device='cpu', exp=None,
                  user_subset=None):
    """ Test a GNN model
    Args:
        model (GNN): Model to test.
        graph (RecGraphData): graph we train on.
        test_ds (Dataset): Test dataset.
        epoch (int): Epoch.
        metrics (List[torchmetrics.Metric]): Metrics to print.
        device (str): Device.
        exp (Experiment | None): Experiment to save results.
        user_subset (Tensor[long]): Subset of users to test on.
    """
    metrics = metrics or []

    # Eval mode
    model.eval()

    # Iterate over batches
    with torch.no_grad():  # no need to compute gradients
        top = time()
        out = model.get_embedding(graph).to(device)

        print(f"Time for inference on one user: {time() - top:.2f} seconds")
        preds = torch.matmul(out[:graph.num_users],
                             out[graph.num_users: graph.num_users+graph.num_items].T)
        print(f"Time for inference on all users: {time() - top:.2f} seconds")
        target = test_ds.val_adj_matrix.to(device)
        exclude = test_ds.train_adj_matrix.to(device)
        # first_inter = test_ds.first_val_interactions.to(device)

        if user_subset is not None:
            preds = preds[user_subset]
            target = target[user_subset]
            exclude = exclude[user_subset]
            # first_inter = first_inter[user_subset]
        first_inter = None

        # Update metrics
        for met in metrics:
            met.update(preds, target, exclude, first_inter)
        for met in metrics:
            print(met.name, met.compute().cpu().item())

        # Save metrics
        if exp:
            exp.on_epoch_end(epoch, metrics)
            exp.save()

    # Free memory
    # del test_ds.val_adj_matrix
    # del test_ds.train_adj_matrix


def __test_fn_cf(model: CF, graph: RecGraphData, test_ds, epoch: int,
                 metrics: Union[List[torchmetrics.Metric], None] = None,
                 device='cpu', exp=None,
                 user_subset=None):
    """ Test a CF model
    Args:
        model (CF): Model to test.
        graph (RecGraphData): Graph.
        test_ds (Dataset): Test dataset.
        epoch (int): Epoch.
        metrics (List[torchmetrics.Metric]): Metrics to print.
        device (str): Device.
        exp (Experiment): Experiment to save results.
        user_subset (Tensor[long]): Subset of users to test.
    """
    metrics = metrics or []

    # Eval mode
    model.eval()

    # Sample new negative items
    test_ds.compute_adj_matrix()
    num_users = test_ds.train_graph.num_users
    num_items = test_ds.train_graph.num_items
    # test_ds.compute_first_test_interactions()

    # Iterate over batches
    with torch.no_grad():
        preds = []
        bs = 2
        items_id = torch.arange(num_users, num_users + num_items, dtype=torch.long, device="cuda:1")
        for u in tqdm(range(0, num_users, bs)):
            # user_id = torch.tensor([u] * num_items, dtype=torch.long, device="cuda")
            user_ids = torch.arange(u, min(u+bs, num_users), dtype=torch.long, device="cuda:1").tile((num_items, 1)).T.flatten()
            items_ids = items_id.repeat(min(bs, num_users-u))
            for p in torch.split(model(graph, user_ids, items_ids).to(device), num_items):
                preds.append(p)
        preds = torch.stack(preds, 0)
        target = test_ds.val_adj_matrix.to(device)
        exclude = test_ds.train_adj_matrix.to(device)
        first_inter = None  # test_ds.first_val_interactions.to(device)

        if user_subset is not None:
            preds = preds[user_subset]
            target = target[user_subset]
            exclude = exclude[user_subset]

        # Update metrics
        for met in metrics:
            met.update(preds, target, exclude, first_inter)
        for m in metrics:
            print(m.name, m.compute())

        # Save metrics
        if exp:
            exp.on_epoch_end(epoch, metrics)
            exp.save()

    # Free memory
    # del test_ds.val_adj_matrix
    # del test_ds.train_adj_matrix


def test_fn(model: Union[CF, GNN], graph: RecGraphData, test_ds, epoch: int,
            metrics: Union[List[torchmetrics.Metric], None] = None,
            device='cpu', exp=None,
            user_subset=None):
    """ Test a model
    Args:
        model (GNN | CF): Model to test.
        graph (RecGraphData): Graph.
        test_ds (Dataset): Test dataset.
        epoch (int): Epoch.
        metrics (List[torchmetrics.Metric]): Metrics to print.
        device (str): Device.
        exp (Experiment): Experiment to save results.
        user_subset (Tensor[long]): Subset of users to test.
    """
    if isinstance(model, GNN):
        return __test_fn_gnn(model, graph, test_ds, epoch, metrics, device, exp, user_subset)
    if isinstance(model, CF):
        return __test_fn_cf(model, graph, test_ds, epoch, metrics, device, exp, user_subset)
    else:
        raise ValueError("Model must be either GNN or CF")
