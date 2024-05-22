import logging
from typing import Dict, List

import torch
from jaxtyping import Bool, Float, Integer
from torchmetrics import Metric

# from torchmetrics import RetrievalMRR, RetrievalHitRate, RetrievalRPrecision


logger = logging.getLogger(__name__)


def get_dtype_and_device(t: torch.Tensor) -> Dict[str, object]:
    return {"dtype": t.dtype, "device": t.device}


class RetrievalMetrics(Metric):
    """Collection of retrieval metrics in a single class."""

    is_differentiable = False
    higher_is_better = None
    full_state_update = False
    count: Integer[torch.Tensor, ""]  # noqa
    ranks_numerator: Integer[torch.Tensor, ""]  # noqa
    reciprocal_ranks_numerator: Float[torch.Tensor, ""]  # noqa
    r_precision_numerator: Float[torch.Tensor, ""]  # noqa
    empty_targets: Integer[torch.Tensor, ""]  # noqa

    def __init__(self, ks_for_hits: List[int] = None, **kwargs):
        """Initialize the metric.

        Args:
            ks_for_hits: list of k values for which to compute hits@k
            kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        # We use a list to store all reciprocal ranks for each update
        # States
        # Ranks
        self.add_state(
            "ranks_numerator", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "reciprocal_ranks_numerator",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        # Count
        self.add_state(
            "count", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        if ks_for_hits is None:
            ks_for_hits = [1, 3, 5, 10]
        # Hits
        self.ks_for_hits = ks_for_hits
        for k in ks_for_hits:
            self.add_state(
                f"hits_at_{k}", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
        # R-precision
        self.add_state(
            "r_precision_numerator",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        # Fraction of empty targets
        self.add_state(
            "empty_targets", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    @property
    def metrics_info(self) -> Dict[str, str]:
        """Returns the names and lower the better or higher the better for each metric."""
        return {
            "reciprocal_rank": "higher",
            "rank": "lower",
            "r_precision": "higher",
            "empty_targets": "lower",
            **{f"hits_at_{k}": "higher" for k in self.ks_for_hits},
        }

    def update(
        self,
        preds: Float[
            torch.Tensor,
            "batch_size num_objects",  # noqa
        ],
        target: Bool[torch.Tensor, "batch_size num_objects"],  # noqa
    ):
        # sort the scores in descending and keep the indices
        # then use the indices to sort the target to the same order
        # finally find the first two index of the target that is True by using argmax
        # To use argmax we need to convert the True to 1 and False to 0 using long
        # add 1 because ranks start from 1
        assert preds.dim() == 2
        assert target.dim() == 2
        assert preds.shape == target.shape
        assert target.dtype == torch.bool
        sorted_indices = preds.argsort(dim=-1, descending=True)
        sorted_target = target.gather(-1, sorted_indices)
        # handle empty targets
        not_empty_target_row = ~(target.to(dtype=torch.long).sum(dim=-1) == 0)
        self.empty_targets += (~not_empty_target_row).sum()
        ranks: Integer[torch.Tensor, "batch"] = (  # noqa
            sorted_target.to(torch.long).argmax(dim=-1)
        ) + 1  # shape (batch_size,)
        # only sum over non-empty targets
        self.ranks_numerator += ranks[not_empty_target_row].sum()
        self.reciprocal_ranks_numerator += (
            1.0 / ranks[not_empty_target_row]
        ).sum()
        self.count += not_empty_target_row.sum()
        for k in self.ks_for_hits:
            hits_at_k = getattr(self, f"hits_at_{k}")
            hits = (ranks[not_empty_target_row] <= k).sum()
            setattr(self, f"hits_at_{k}", hits_at_k + hits)
        # r-precision
        col_indices = (
            torch.arange(preds.size(1), device=target.device)
            .unsqueeze(0)
            .expand_as(preds)
        )  # shape (batch_size, num_objects)
        _R = target.sum(dim=-1, keepdim=True)  # shape (batch_size, 1)
        mask = col_indices < _R  # shape (batch_size, num_objects)
        _r = (sorted_target * mask).sum(dim=-1)  # shape (batch_size,)
        r_precision = _r[not_empty_target_row] / (
            _R.squeeze(-1)[not_empty_target_row]
        )
        self.r_precision_numerator += r_precision.sum()

    def compute(
        self,
    ) -> Dict[
        str,
        Float[torch.Tensor, ""],  # noqa
    ]:
        if self.count.item() == 0:
            logger.warning("compute() called before update()")
            return {
                "reciprocal_rank": torch.tensor(
                    0.0, **get_dtype_and_device(self.ranks_numerator)
                ),
                "rank": torch.tensor(
                    0.0, **get_dtype_and_device(self.ranks_numerator)
                ),
                "r_precision": torch.tensor(
                    0.0, **get_dtype_and_device(self.r_precision_numerator)
                ),
                "empty_targets": torch.tensor(0.0, **get_dtype_and_device(self.ranks_numerator)),
                **{
                    f"hits_at_{k}": torch.tensor(
                        0.0, **get_dtype_and_device(self.ranks_numerator)
                    )
                    for k in self.ks_for_hits
                },
            }

        hits_at_k = {
            f"hits_at_{k}": getattr(self, f"hits_at_{k}") / self.count
            for k in self.ks_for_hits
        }
        return {
            "reciprocal_rank": self.reciprocal_ranks_numerator / self.count,
            "rank": self.ranks_numerator / self.count,
            "r_precision": self.r_precision_numerator / self.count,
            "empty_targets": self.empty_targets / self.count,
            **hits_at_k,
        }