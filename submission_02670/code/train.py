"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    preds = []
    for start in range(0, X.shape[0], batch_size):
        xb = X[start:start + batch_size]
        preds.append(model(xb).argmax(dim=1))
    if not preds:
        return X.new_empty((0,), dtype=torch.long)
    return torch.cat(preds, dim=0)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval()
    total_loss = 0.0
    total_n = 0
    pred_parts = []
    y_parts = []

    for xb, yb in iterate_batches(X, y, batch_size=batch_size, shuffle=False):
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)
        total_loss += float(loss.item()) * xb.size(0)
        total_n += xb.size(0)
        pred_parts.append(logits.argmax(dim=1))
        y_parts.append(yb)

    pred = torch.cat(pred_parts) if pred_parts else torch.empty(0, dtype=torch.long, device=X.device)
    y_true = torch.cat(y_parts) if y_parts else torch.empty(0, dtype=torch.long, device=X.device)
    loss = total_loss / max(1, total_n)
    acc = float((pred == y_true).float().mean().item()) if total_n > 0 else 0.0

    cm = np.zeros((7, 7), dtype=np.int64)
    if total_n > 0:
        np.add.at(cm, (y_true.cpu().numpy(), pred.cpu().numpy()), 1)
    macro_f1 = macro_f1_from_confusion(cm)
    return {"loss": float(loss), "acc": acc, "macro_f1": macro_f1}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction="mean")
    if loss_name == "mse":
        target = F.one_hot(y, num_classes=logits.size(1)).to(dtype=logits.dtype)
        return F.mse_loss(logits, target, reduction="mean")
    raise ValueError(f"Unsupported loss_name={loss_name!r}; expected 'ce' or 'mse'")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    set_seed(cfg["seed"])
    device = data["X_tr"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    assert hidden in EXPECTED_PARAMS, f"Unsupported architecture {hidden}; expected one of {sorted(EXPECTED_PARAMS)}"
    assert count_params(MLP(hidden=hidden, dropout=cfg.get("dropout", 0.0), init=cfg.get("init", "he"))) == EXPECTED_PARAMS[hidden]

    model = MLP(hidden=hidden, dropout=cfg.get("dropout", 0.0), init=cfg.get("init", "he")).to(device)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0), momentum=cfg.get("momentum", 0.9))

    scaler = None
    if cfg.get("precision") == "fp16":
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda") if device.type == "cuda" else None

    step0_loss = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"], batch_size=cfg["batch"])["loss"]

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "grad_norm": [], "epoch_time_s": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    diverged = False
    start_total = time.perf_counter()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        grad_norms = []
        epoch_loss_sum = 0.0
        epoch_n = 0
        generator = torch.Generator(device=device)
        generator.manual_seed(cfg["seed"] + epoch)

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], cfg["batch"], generator=generator, shuffle=True):
            if cfg.get("precision") in {"fp16", "bf16"}:
                dtype = torch.float16 if cfg["precision"] == "fp16" else torch.bfloat16
                with torch.autocast(device_type=device.type, dtype=dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            if not torch.isfinite(loss):
                diverged = True
                break

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if scaler is not None:
                scaler.unscale_(optimizer)

            grad_norm = clip_gradients(model.parameters(), cfg.get("clip_norm"))
            grad_norms.append(grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            epoch_loss_sum += float(loss.detach().item()) * xb.size(0)
            epoch_n += xb.size(0)

        if diverged:
            break

        epoch_time = time.perf_counter() - start_total
        train_metrics = evaluate(model, data["X_tr"], data["y_tr"], loss_name=cfg["loss"], batch_size=cfg["batch"])
        val_metrics = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"], batch_size=cfg["batch"])
        grad_norm_mean = float(np.mean(grad_norms)) if grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(grad_norm_mean)
        history["epoch_time_s"].append(epoch_time)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if not history["epoch"]:
        history["epoch"].append(1)
        history["train_loss"].append(step0_loss)
        history["val_loss"].append(step0_loss)
        history["val_acc"].append(0.0)
        history["val_macro_f1"].append(0.0)
        history["grad_norm"].append(0.0)
        history["epoch_time_s"].append(0.0)
        best_epoch = 1
        best_val_loss = step0_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    best_idx = history["val_loss"].index(min(history["val_loss"]))
    val_acc_best = history["val_acc"][best_idx]
    val_f1_best = history["val_macro_f1"][best_idx]

    peak_mem_MB = 0.0
    if device.type == "cuda":
        peak_mem_MB = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": best_epoch,
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "val_acc": float(val_acc_best),
        "val_macro_f1": float(val_f1_best),
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0,
        "peak_mem_MB": peak_mem_MB,
        "diverged": bool(diverged),
    }

    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    out = pd.DataFrame({"row_id": np.asarray(row_id).astype(np.int64), "pred": np.asarray(preds).astype(np.int64)})
    out.to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    hidden = tuple(cfg.get("hidden", (256, 128)))
    device = data["X_eval"].device
    model = MLP(hidden=hidden, dropout=cfg.get("dropout", 0.0), init=cfg.get("init", "he")).to(device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"], batch_size=cfg.get("batch", 512))
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script = os.path.join(repo_root, "scripts", "evaluate.py")
    out_json = os.path.splitext(pred_path)[0] + "_result.json"
    subprocess.run([sys.executable, script, "--pred", pred_path, "--out", out_json], cwd=repo_root, check=True)
