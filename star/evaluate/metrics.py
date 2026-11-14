"""CMC and trapezoidal AP using the existing University-1652 protocol."""

import gc

import numpy as np
import torch
from tqdm import tqdm

from .features import extract_features


def compute_mAP(index, good_index, junk_index):
    index = np.asarray(index).reshape(-1)
    good = np.asarray(good_index).reshape(-1)
    junk = np.asarray(junk_index).reshape(-1)
    cmc = torch.zeros(len(index), dtype=torch.int32)
    if not len(index):
        raise ValueError("The gallery contains no samples.")
    good = good[~np.isin(good, junk)]
    if not len(good):
        cmc[0] = -1
        return 0.0, cmc
    ranking = index[~np.isin(index, junk)]
    rows = np.flatnonzero(np.isin(ranking, good))
    if len(rows) != len(good):
        raise ValueError("The ranking does not contain every positive gallery item.")
    cmc[int(rows[0]) :] = 1
    ap = 0.0
    for i, row in enumerate(rows):
        precision = (i + 1) / (row + 1)
        previous_precision = i / row if row else 1.0
        ap += (previous_precision + precision) / (2 * len(good))
    return ap, cmc


def eval_query(qf, ql, gf, gl):
    scores = (gf @ qf).reshape(-1).detach().cpu().numpy()
    ranking = np.argsort(-scores, kind="stable")
    labels = np.asarray(gl).reshape(-1)
    good = (
        np.flatnonzero((labels == ql) & (labels >= 0))
        if ql >= 0
        else np.array([], dtype=int)
    )
    return compute_mAP(ranking, good, np.flatnonzero(labels < 0))


def compute_metrics(
    query_features,
    query_ids,
    gallery_features,
    gallery_ids,
    ranks=(1, 5, 10),
    step_size=1000,
    verbose=True,
):
    if step_size < 1 or any(rank < 1 for rank in ranks):
        raise ValueError("step_size and recall ranks must be positive.")
    if query_features.ndim != 2 or gallery_features.ndim != 2:
        raise ValueError("Features must be two-dimensional descriptor matrices.")
    if query_features.size(1) != gallery_features.size(1):
        raise ValueError("Query and gallery descriptor dimensions differ.")
    query_count, gallery_count = len(query_features), len(gallery_features)
    ql = torch.as_tensor(query_ids).reshape(-1).cpu().numpy()
    gl = torch.as_tensor(gallery_ids).reshape(-1).cpu().numpy()
    if query_count == 0 or gallery_count == 0:
        raise ValueError("Both query and gallery must contain samples.")
    if len(ql) != query_count or len(gl) != gallery_count:
        raise ValueError("Descriptor and sample ID counts differ.")
    if (
        not torch.isfinite(query_features).all()
        or not torch.isfinite(gallery_features).all()
    ):
        raise ValueError("Non-finite retrieval descriptors.")

    cmc = torch.zeros(gallery_count, dtype=torch.int64)
    ap_sum, matched = 0.0, 0
    junk = np.flatnonzero(gl < 0)
    gallery_features = gallery_features.to(query_features.device, dtype=torch.float32)
    batches = range(0, query_count, step_size)
    for start in tqdm(batches, desc="Ranking", disable=not verbose):
        scores = (
            (query_features[start : start + step_size].float() @ gallery_features.T)
            .detach()
            .cpu()
            .numpy()
        )
        rankings = np.argsort(-scores, axis=1, kind="stable")
        for offset, ranking in enumerate(rankings):
            label = ql[start + offset]
            good = (
                np.flatnonzero((gl == label) & (gl >= 0))
                if label >= 0
                else np.array([], dtype=int)
            )
            ap, query_cmc = compute_mAP(ranking, good, junk)
            if query_cmc[0] < 0:
                continue
            cmc += query_cmc
            ap_sum += ap
            matched += 1
    # Match the previous denominator: queries without positives count as failures.
    cmc = cmc.double() / query_count
    top_percent_rank = min(gallery_count, max(1, round(gallery_count * 0.01)))
    recalls = {
        str(rank): float(cmc[min(rank, gallery_count) - 1])
        for rank in sorted(set(ranks) | {1})
    }
    return {
        "recall_at_k": recalls,
        "mAP": ap_sum / query_count,
        "recall_at_1_percent": float(cmc[top_percent_rank - 1]),
        "top_1_percent_rank": top_percent_rank,
        "query_count": query_count,
        "gallery_count": gallery_count,
        "matched_query_count": matched,
        "queries_without_positive": query_count - matched,
        "ignored_gallery_count": len(junk),
    }


def evaluate_retrieval(
    config,
    model,
    query_loader,
    gallery_loader,
    ranks=(1, 5, 10),
    step_size=1000,
    cleanup=True,
    query_sequence=False,
    return_metrics=False,
):
    query_features, query_ids = extract_features(
        config, model, query_loader, sequence=query_sequence
    )
    gallery_features, gallery_ids = extract_features(config, model, gallery_loader)
    metrics = compute_metrics(
        query_features,
        query_ids,
        gallery_features,
        gallery_ids,
        ranks=ranks,
        step_size=step_size,
        verbose=getattr(config, "verbose", True),
    )
    text = [
        f'Recall@{rank}: {100*metrics["recall_at_k"][str(rank)]:.4f}' for rank in ranks
    ]
    text += [
        f'Recall@top1% (K={metrics["top_1_percent_rank"]}): {100*metrics["recall_at_1_percent"]:.4f}',
        f'mAP: {100*metrics["mAP"]:.4f}',
    ]
    print(" - ".join(text), flush=True)
    if metrics["queries_without_positive"]:
        print(
            f'Queries without gallery positives: {metrics["queries_without_positive"]}/{metrics["query_count"]}',
            flush=True,
        )
    if cleanup:
        del query_features, query_ids, gallery_features, gallery_ids
        gc.collect()
    return (
        metrics
        if return_metrics
        else torch.tensor(metrics["recall_at_k"]["1"], dtype=torch.float32)
    )
