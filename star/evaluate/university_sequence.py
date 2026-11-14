"""Sequence-query evaluation with the shared retrieval metrics."""

from .metrics import compute_mAP, eval_query, evaluate_retrieval


def evaluate(
    config,
    model,
    query_loader,
    gallery_loader,
    ranks=(1, 5, 10),
    step_size=1000,
    cleanup=True,
    return_metrics=False,
):
    return evaluate_retrieval(
        config,
        model,
        query_loader,
        gallery_loader,
        ranks=ranks,
        step_size=step_size,
        cleanup=cleanup,
        query_sequence=True,
        return_metrics=return_metrics,
    )
