""" Evaluation functions for cold start recommendation. """
import torch
import pickle
from RecSys.cold_start.data import ColdStartData
from RecSys.cold_start.models import ColdStartModel
from typing import List, Iterable


def final_evaluation(model: ColdStartModel, data: ColdStartData, metrics: List, nb_inter_range: Iterable, res_path: str, device="cpu"):
    """ Evaluate the model on cold start recommendation.

    The model is evaluated on recommending the next iteractions of the new users knowing its first `m` interactions.

    Args:
        model (ColdStartModel): Model to evaluate.
        data (ColdStartData): Data to evaluate on.
        metrics (List): Metrics to evaluate.
        nb_inter_range (Iterable): Range of the number `m` of interactions considered known.
        res_path (str): Path to save the results.
        device (str): Device.
    """
    results = dict()

    with torch.no_grad():
        for nb_inter in nb_inter_range:
            results[nb_inter] = dict()
            print(f"Metrics for {nb_inter} interactions known per user:")
            pre_graph, post_graph, test_ds, users_id = data.get_test_data(nb_inter, max(nb_inter_range))
            preds = model(data, pre_graph, users_id).to(device)
            test_ds.compute_adj_matrix()
            del test_ds.train_adj_matrix
            val_adj_matrix = test_ds.val_adj_matrix[users_id].to(device)

            exclude = None
            first_inter = None

            for metric in metrics:
                metric.update(preds, val_adj_matrix, exclude, first_inter)
                print(f"\t{metric.name}: {metric.compute().cpu().item()}")
                results[nb_inter][metric.name] = metric.metric.cpu().numpy()

    with open(res_path, "wb") as f:
        pickle.dump(results, f)
