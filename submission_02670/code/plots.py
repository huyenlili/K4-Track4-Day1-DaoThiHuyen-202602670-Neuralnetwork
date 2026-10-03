"""plots.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    history = result.get("history", {})
    cfg = result.get("cfg", {})
    epochs = history.get("epoch", [])
    if not epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax1, ax2, ax3 = axes

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    ax1.plot(epochs, train_loss, label="train_loss", color="tab:blue")
    ax1.plot(epochs, val_loss, label="val_loss", color="tab:orange")
    best_epoch = result.get("summary", {}).get("best_epoch", epochs[0])
    ax1.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax1.set_title(f"{cfg.get('exp_id', 'exp')} — loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()

    val_acc = history.get("val_acc", [])
    val_macro_f1 = history.get("val_macro_f1", [])
    ax2.plot(epochs, val_acc, label="val_acc", color="tab:green")
    ax2.plot(epochs, val_macro_f1, label="val_macro_f1", color="tab:red")
    ax2.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax2.set_title("validation metrics")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score")
    ax2.legend()

    grad_norm = history.get("grad_norm", [])
    ax3.plot(epochs, grad_norm, label="grad_norm", color="tab:purple")
    ax3.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax3.set_title("gradient norm")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("norm")
    ax3.legend()

    title = (
        f"exp_id={cfg.get('exp_id', 'unknown')} | "
        f"opt={cfg.get('optimizer', 'n/a')} | lr={cfg.get('lr', 'n/a')} | "
        f"batch={cfg.get('batch', 'n/a')} | hidden={cfg.get('hidden', 'n/a')}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        return
    plt.figure(figsize=(10, 6))
    for result in results:
        history = result.get("history", {})
        epochs = history.get("epoch", [])
        values = history.get(metric, [])
        if not epochs or not values:
            continue
        label = result.get("cfg", {}).get("exp_id", "unknown")
        plt.plot(epochs, values, label=label)
    plt.title(title or f"Comparison on {metric}")
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
