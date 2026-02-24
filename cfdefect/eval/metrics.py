from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support

from ..utils.rle import rle_decode

def _safe_auc(y_true: List[int], y_score: List[float]) -> float:
    # roc_auc requires both classes present
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

def dice_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.shape != gt.shape:
        import cv2
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    p = int(pred.sum())
    g = int(gt.sum())
    dice = (2 * inter) / (p + g + 1e-9)
    iou = inter / (union + 1e-9)
    return float(dice), float(iou)

def f1_set(pred_set: List[int], gt_set: List[int]) -> float:
    ps = set(int(x) for x in pred_set)
    gs = set(int(x) for x in gt_set)
    if not ps and not gs:
        return 1.0
    if not ps and gs:
        return 0.0
    if ps and not gs:
        return 0.0
    tp = len(ps & gs)
    fp = len(ps - gs)
    fn = len(gs - ps)
    return float((2 * tp) / (2 * tp + fp + fn + 1e-9))

def evaluate(gt_records: List[Dict[str, Any]],
             pred_records: List[Dict[str, Any]],
             load_mask_fn) -> Dict[str, Any]:
    """
    gt_records: list from meta.jsonl
    pred_records: list from pred.jsonl
    load_mask_fn: callable(path)->np.ndarray for GT masks and optionally predictions
    """
    pred_by_id = {p["id"]: p for p in pred_records}

    y_true = []
    y_score = []

    loc_dice = []
    loc_iou = []

    cls_acc = []
    cls_top3 = []

    attr_f1s = []

    missing_pred = 0

    for g in gt_records:
        gid = g["id"]
        is_def = 1 if g.get("is_defective", False) else 0
        y_true.append(is_def)

        p = pred_by_id.get(gid)
        if p is None:
            missing_pred += 1
            y_score.append(0.0)
            continue
        score = float(p.get("score", 0.0))
        y_score.append(score)

        # Localization
        if is_def == 1:
            gt_mask = load_mask_fn(g["mask_path"])
            # predicted mask can be rle or mask_path
            if "mask_rle" in p and p["mask_rle"] is not None:
                pred_mask = rle_decode(p["mask_rle"])
            elif "mask_path" in p and p["mask_path"]:
                pred_mask = load_mask_fn(p["mask_path"])
            else:
                pred_mask = np.zeros_like(gt_mask)
            d, iou = dice_iou(pred_mask, gt_mask)
            loc_dice.append(d)
            loc_iou.append(iou)

            # Classification
            gt_types = [dct.get("type") for dct in g.get("defects", []) if dct.get("type")]
            pred_types = p.get("types", [])
            if isinstance(pred_types, str):
                pred_types = [pred_types]
            top1 = 1.0 if (pred_types and gt_types and pred_types[0] in gt_types) else 0.0
            top3 = 1.0 if (gt_types and any(t in gt_types for t in (pred_types[:3] if pred_types else []))) else 0.0
            cls_acc.append(top1)
            cls_top3.append(top3)

            # Attribution
            gt_paths: List[int] = []
            for dct in g.get("defects", []):
                gt_paths.extend(dct.get("path_ids", []) or [])
            pred_paths = p.get("path_ids", []) or []
            attr_f1s.append(f1_set(pred_paths, gt_paths))

    # Detection metrics
    auc = _safe_auc(y_true, y_score)
    # threshold at 0.5 for now; you can also compute best F1
    y_pred = [1 if s >= 0.5 else 0 for s in y_score]
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec, rec, f1pr, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    out = {
        "n": len(gt_records),
        "missing_pred": int(missing_pred),
        "detection": {
            "auroc": auc,
            "f1@0.5": f1,
            "precision@0.5": float(prec),
            "recall@0.5": float(rec),
        },
        "localization": {
            "dice": float(np.mean(loc_dice)) if loc_dice else float("nan"),
            "miou": float(np.mean(loc_iou)) if loc_iou else float("nan"),
        },
        "classification": {
            "top1": float(np.mean(cls_acc)) if cls_acc else float("nan"),
            "top3": float(np.mean(cls_top3)) if cls_top3 else float("nan"),
        },
        "attribution": {
            "path_f1": float(np.mean(attr_f1s)) if attr_f1s else float("nan"),
        }
    }
    return out
