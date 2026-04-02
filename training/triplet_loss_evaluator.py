import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator

from tasks import TripletLoss


class TripletLossEvaluator(SentenceEvaluator):
    """
    Replicates the margin-based TripletLoss from pl_training.py on an eval split.
    Uses TripletLoss from tasks.py directly (default: l2-norm, margin=1.0).
    For multiple negatives per row, expands into K triplets and averages loss across all,
    matching pl_training's pairwise evaluation semantics.
    Returns negative mean loss so higher = better (required by SentenceTransformerTrainer).
    """

    def __init__(
        self,
        eval_ds,
        name: str = "",
        batch_size: int = 64,
        margin: float = 1.0,
        distance: str = "l2-norm",
    ):
        self.name = name
        self.batch_size = batch_size
        self.loss_fn = TripletLoss(margin=margin, distance=distance, reduction="none")

        self.anchors = eval_ds["anchor"]
        self.positives = eval_ds["positive"]
        # Collect all negative columns in order: "negative" (K=1) or "negative_1", "negative_2", ...
        if "negative" in eval_ds.column_names:
            self.neg_cols = [eval_ds["negative"]]
        else:
            k = 1
            self.neg_cols = []
            while f"negative_{k}" in eval_ds.column_names:
                self.neg_cols.append(eval_ds[f"negative_{k}"])
                k += 1

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        q_emb = torch.tensor(model.encode(self.anchors, batch_size=self.batch_size, show_progress_bar=False))
        p_emb = torch.tensor(model.encode(self.positives, batch_size=self.batch_size, show_progress_bar=False))

        # Compute per-sample loss for each negative column, then average across K
        all_losses = torch.stack([
            self.loss_fn(q_emb, p_emb, torch.tensor(model.encode(negs, batch_size=self.batch_size, show_progress_bar=False)))
            for negs in self.neg_cols
        ])  # shape: (K, N)
        mean_loss = all_losses.mean().item()

        print(f"[{self.name}] triplet_loss={mean_loss:.4f} (epoch={epoch}, steps={steps})")
        self.primary_metric = f"{self.name}_triplet_loss"
        return {self.primary_metric: mean_loss}
