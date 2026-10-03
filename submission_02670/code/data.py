"""data.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    with np.load(f"{processed_dir}/train.npz") as train, np.load(f"{processed_dir}/eval.npz") as eval_data:
        X_train_full = train["X"]
        y_train_full = train["y"]
        X_eval = eval_data["X"]
        y_eval = eval_data["y"]
        eval_row_id = eval_data["row_id"]

    assert X_train_full.shape[1] == 54, f"X_train_full phải có 54 feature, nhận {X_train_full.shape}"
    assert X_eval.shape[1] == 54, f"X_eval phải có 54 feature, nhận {X_eval.shape}"
    assert X_train_full.dtype == np.float32, f"X_train_full phải float32, nhận {X_train_full.dtype}"
    assert X_eval.dtype == np.float32, f"X_eval phải float32, nhận {X_eval.dtype}"
    assert y_train_full.dtype == np.int64, f"y_train_full phải int64, nhận {y_train_full.dtype}"
    assert y_eval.dtype == np.int64, f"y_eval phải int64, nhận {y_eval.dtype}"
    assert y_train_full.shape[0] == X_train_full.shape[0]
    assert y_eval.shape[0] == X_eval.shape[0]
    assert eval_row_id.dtype == np.int64

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=val_fraction,
        stratify=y,
        random_state=seed,
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    mean = X_tr[:, :N_NUMERIC].mean(axis=0)
    std = X_tr[:, :N_NUMERIC].std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X_copy = X.copy()
    X_copy[:, :N_NUMERIC] = (X_copy[:, :N_NUMERIC] - mean) / std
    return X_copy.astype(np.float32)


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction=val_fraction, seed=seed)
    mean, std = fit_standardizer(X_tr)

    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    X_tr = torch.tensor(X_tr, device=device, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, device=device, dtype=torch.int64)
    X_val = torch.tensor(X_val, device=device, dtype=torch.float32)
    y_val = torch.tensor(y_val, device=device, dtype=torch.int64)
    X_eval = torch.tensor(X_eval, device=device, dtype=torch.float32)
    y_eval = torch.tensor(y_eval, device=device, dtype=torch.int64)

    majority_label = int(np.bincount(y_train_full).argmax())
    majority_acc = float((y_val.cpu().numpy() == majority_label).mean())
    print(f"X_tr={tuple(X_tr.shape)}, X_val={tuple(X_val.shape)}, X_eval={tuple(X_eval.shape)}")
    print(f"majority-class baseline on val (label {majority_label}): acc = {majority_acc:.4f}")

    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_val": X_val,
        "y_val": y_val,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "eval_row_id": eval_row_id,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    N = X.shape[0]
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)

    for start in range(0, N, batch_size):
        idx = perm[start:start + batch_size]
        yield X[idx], y[idx]
