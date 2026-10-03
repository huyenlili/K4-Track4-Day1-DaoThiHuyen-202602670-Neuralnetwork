# Báo cáo Lab Day 1 — Huyen Li — 02670

## 1. Thiết lập

- Môi trường ghi trong output của `lab.ipynb`: Python 3.13.15, PyTorch 2.11.0+cpu, CPU (không có CUDA GPU).
- Dữ liệu Forest CoverType: train 464 809 mẫu và eval 116 203 mẫu theo `split_metadata.csv`. Từ train, tách validation phân tầng 20% với seed 42: 371 847 mẫu train và 92 962 mẫu validation. Mười đặc trưng số được chuẩn hóa theo thống kê của phần train; 44 đặc trưng nhị phân giữ nguyên.
- Mô hình: M-base, 54 → 256 → 128 → 7, ReLU ở các lớp ẩn, 47 879 tham số. Loss cross-entropy, optimizer SGD với momentum (theo `DEFAULT_CFG` trong notebook), batch 512, 20 epoch, He initialization. Learning rate được chọn bằng validation: `lr=0.1`.
- Accuracy tham chiếu khi luôn đoán lớp đa số trên validation: 0.4876.
- Các thí nghiệm đã chạy: so sánh CE-vs-MSE, sweep learning rate, baseline với ba seed (0, 1, 2), và dropout 0.3. Không chạy optimizer khác SGD+momentum, clipping, mixed precision hay so sánh initialization; các mục đó không được kết luận trong báo cáo này.

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7); kiểm tra forward dùng (8, 54) → (8, 7) |
| Loss ban đầu trên validation | 2.3776; tham chiếu ln 7 = 1.9459 |
| Quá khớp 20 mẫu | Cross-entropy giảm từ 1.9668 xuống 0.000266 sau 500 bước |
| Gradient của từng tham số | Tất cả 6 tensor tham số có gradient khác 0; grad norm lần lượt ≈ 0.5823, 0.3834, 2.3718, 0.4982, 2.0784, 0.6061 |
| Baseline | 3 seed: `base-s0`, `base-s1`, `base-s2` |
| Val accuracy, trung bình ± độ lệch chuẩn mẫu | 0.9088 ± 0.0013 |
| Val macro-F1, trung bình ± độ lệch chuẩn mẫu | 0.8602 ± 0.0013 |

Ngưỡng tham khảo 2σ giữa các seed cho val macro-F1 là 0.0026. Chỉ có ba seed và cùng một cấu hình baseline, nên ngưỡng này chỉ là ước lượng sơ bộ, không phải kiểm định thống kê.

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát — CE vs MSE

**Dự đoán:** cross-entropy sẽ đạt macro-F1 validation cao hơn MSE, vì cross-entropy áp dụng softmax và trực tiếp tối ưu xác suất cho lớp đúng, trong khi MSE trên logits có thể tạo tín hiệu gradient kém phù hợp hơn cho phân loại nhiều lớp.

Hai run dùng cùng seed 0, split, kiến trúc M-base, SGD+momentum, `lr=0.1`, batch 512, 20 epoch, He initialization và không dropout; chỉ thay hàm loss.

| Experiment | Loss | Best epoch | Val loss (theo loss tương ứng) | Val accuracy | Val macro-F1 |
|---|---|---:|---:|---:|---:|
| `loss-ce-s0` | Cross-entropy | 19 | 0.23022 | 0.90763 | 0.85701 |
| `loss-mse-s0` | MSE trên logits và one-hot, mean trên mọi phần tử | 19 | 0.02982 | 0.86869 | 0.73456 |

![So sánh CE và MSE trên validation](figures/compare_loss.png)

Có thể chạy lại thí nghiệm MSE và xuất JSON/biểu đồ so sánh từ thư mục gốc repo bằng lệnh `python submission_02670/code/run_loss_comparison.py`. Script giữ nguyên cấu hình baseline `base-s0`, chỉ đổi loss sang MSE; nó cần `output/data/processed/` và kết quả CE seed 0 trong `output/results/`.

Val macro-F1 của MSE thấp hơn CE 0.12245, lớn hơn nhiều ngưỡng nhiễu 2σ = 0.0026 ước lượng từ ba seed baseline. Kết luận trong phạm vi thí nghiệm này là CE tốt hơn rõ rệt cho cấu hình đã thử. Không so sánh trực tiếp hai giá trị val loss: CE lấy trung bình theo mẫu, còn MSE lấy trung bình theo các phần tử trong ma trận logits/one-hot, nên thang đo và ý nghĩa của chúng khác nhau. CE gradient theo logits là softmax(logits) trừ nhãn one-hot; MSE dùng sai số trực tiếp giữa logits và one-hot, nên thay đổi scale logits cũng làm thay đổi gradient.

### 3.2 Learning rate

**Dự đoán:** learning rate quá nhỏ sẽ học chậm trong 20 epoch; tăng learning rate hợp lý sẽ cải thiện macro-F1 validation.

| Experiment | Learning rate | Val macro-F1 |
|---|---:|---:|
| `lrsweep-0.01` | 0.01 | 0.7719 |
| `lrsweep-0.03` | 0.03 | 0.8264 |
| `lrsweep-0.05` | 0.05 | 0.8455 |
| `lrsweep-0.1` | 0.1 | 0.8594 |

Trong các learning rate đã thử, 0.1 cho val macro-F1 cao nhất và được dùng cho baseline. Đây là kết quả trong phạm vi sweep đã chạy, không chứng minh 0.1 là learning rate tối ưu toàn cục. Các run sweep dùng seed 0 và 20 epoch.

### 3.3 Độ nhiễu giữa seed của baseline

| Experiment | Seed | Val accuracy | Val macro-F1 | Best epoch |
|---|---:|---:|---:|---:|
| `base-s0` | 0 | 0.90835 | 0.85943 | 20 |
| `base-s1` | 1 | 0.91020 | 0.85958 | 19 |
| `base-s2` | 2 | 0.90781 | 0.86172 | 20 |

Độ lệch chuẩn mẫu của val macro-F1 là 0.0013; chênh lệch giữa `base-s2` và trung bình baseline khoảng 0.0015, nhỏ hơn ngưỡng 2σ = 0.0026. Vì vậy không nên xem riêng mức tăng của seed 2 là bằng chứng chắc chắn về cải thiện vượt độ nhiễu.

### 3.4 Dropout

**Dự đoán:** dropout 0.3 có thể giảm khoảng cách train–validation nếu mô hình bị overfit, nhưng có thể làm học khó hơn.

| Experiment | Val macro-F1 | Val accuracy | Final train loss | Final val loss |
|---|---:|---:|---:|---:|
| `base-s0` (dropout 0.0) | 0.85943 | 0.90835 | 0.20746 | 0.22818 |
| `dropout-0.3` | 0.77971 | 0.86913 | 0.31005 | 0.31767 |

Dropout làm khoảng cách train–validation loss giảm từ khoảng 0.0207 xuống 0.0076, nhưng val macro-F1 giảm 0.0805 so với baseline seed 0. Mức giảm này lớn hơn nhiều ngưỡng baseline 2σ = 0.0026. Trong thí nghiệm này, dropout 0.3 gây regularization quá mạnh/underfitting; không có bằng chứng rằng baseline cần dropout.

## 4. Đánh giá cuối trên tập eval

Cấu hình được chọn trong `lab.ipynb` theo val macro-F1 là `base-s2`: SGD với momentum, `lr=0.1`, batch 512, 20 epoch, M-base, dropout 0, seed 2. Kết quả evaluator được in trong notebook:

Các số eval dưới đây là output của evaluator đã lưu trong notebook. Trong thư mục submission hiện chưa có `predictions_eval.csv` và `eval_result.json`, vì vậy chưa thể xem đây là các artifact eval đã được đóng gói/đối chiếu lại.

| Cấu hình | Seed | Val macro-F1 | Eval macro-F1 | Eval accuracy |
|---|---:|---:|---:|---:|
| Baseline tham chiếu `base-s0` | 0 | 0.85943 | 0.86117 | 0.90703 |
| Cấu hình cuối `base-s2` | 2 | 0.86172 | 0.86449 | 0.90591 |

Eval macro-F1 của cấu hình cuối cao hơn baseline seed 0 khoảng 0.00332, trong khi accuracy thấp hơn khoảng 0.00112. Mức chênh macro-F1 eval này chỉ so sánh một seed cuối với một seed baseline; không đủ để kết luận cải thiện có ý nghĩa thống kê. Chênh lệch val–eval của cấu hình cuối khoảng 0.00277.

### 4.1 Phân tích lỗi theo lớp

| Lớp | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 0 | 42 368 | 0.8825 | 0.9263 | 0.9039 |
| 1 | 56 661 | 0.9393 | 0.8967 | 0.9175 |
| 2 | 7 151 | 0.9001 | 0.9070 | 0.9035 |
| 3 | 549 | 0.7575 | 0.8707 | 0.8102 |
| 4 | 1 899 | 0.7509 | 0.7873 | 0.7686 |
| 5 | 3 473 | 0.8191 | 0.8290 | 0.8240 |
| 6 | 4 102 | 0.9022 | 0.9464 | 0.9237 |

Lớp 4 có F1 thấp nhất (0.7686). Trong ma trận nhầm lẫn của output notebook, 317 mẫu lớp 4 bị dự đoán thành lớp 1; đây là nhầm lẫn lớn nhất của lớp 4. Lớp 4 có ít mẫu hơn nhiều so với hai lớp lớn 0 và 1, nên dữ liệu mất cân bằng có thể góp phần gây khó khăn. Một hướng tiếp theo là thử class-weighted cross-entropy và đánh giá lại chỉ bằng validation.

## 5. Trả lời câu hỏi dẫn dắt

1. Chỉ SGD với momentum được dùng trong các lần chạy đã ghi nhận, nên không thể kết luận optimizer nào thắng hoặc so sánh optimizer khi learning rate được chỉnh công bằng. Trong sweep, `lr=0.1` đạt macro-F1 cao nhất trong bốn giá trị đã thử. Ở đối chứng loss cùng cấu hình, CE đạt val macro-F1 0.85701, cao hơn MSE 0.73456 (`loss-ce-s0` và `loss-mse-s0`).
2. Với cùng seed 0, dropout 0.3 giảm khoảng cách train–validation loss nhưng làm val macro-F1 giảm mạnh từ 0.85943 xuống 0.77971. Kết quả này không ủng hộ dùng mức dropout đó cho cấu hình hiện tại; cần thử mức nhỏ hơn hoặc chỉ dùng khi có dấu hiệu overfit rõ.
3. Không có thí nghiệm clipping trong output đã lưu, vì vậy chưa thể kết luận clipping có cứu được run với learning rate cao hay không. Các grad norm được đo ở kiểm tra ban đầu không thay thế cho thí nghiệm clipping.
4. Không có so sánh FP32/FP16/BF16. Notebook chạy trên CPU, nhưng chưa đo mixed precision nên không thể kết luận về tốc độ hoặc độ chính xác của các chế độ này.
5. Thí nghiệm hiện có dùng He initialization; không có đối chứng zero/Xavier. Về nguyên lý, khởi tạo toàn số 0 khiến các neuron cùng lớp nhận gradient giống nhau và khó học các đặc trưng khác nhau; He điều chỉnh độ lớn trọng số cho lớp dùng ReLU. Đây là giải thích lý thuyết, không phải kết luận từ một phép so sánh trong lab này.
6. Nếu loss không giảm sau 2 000 bước, ba kiểm tra đầu tiên là: (i) xác nhận dữ liệu/nhãn đúng shape, đúng kiểu và chuẩn hóa đúng; (ii) kiểm tra forward, loss ban đầu và thử overfit một tập nhỏ để xác nhận model–loss–optimizer có thể học; (iii) kiểm tra gradient có tồn tại, hữu hạn và có độ lớn hợp lý, sau đó rà learning rate/optimizer. Các kiểm tra này phân biệt lỗi dữ liệu, lỗi truyền gradient và bước cập nhật không phù hợp.

## 6. Hạn chế và điều bất ngờ

- Sweep learning rate chỉ dùng seed 0; baseline có ba seed. Số seed còn ít và sweep không bao phủ mọi giá trị learning rate.
- Chỉ thực hiện một mức dropout và một seed cho đối chứng loss; không so sánh optimizer khác, clipping, mixed precision hay initialization.
- Trường `time_per_epoch_s` được code notebook tính bằng trung bình của thời gian tích lũy kể từ đầu run ở cuối mỗi epoch, không phải thời gian riêng từng epoch; vì vậy không dùng trường này để so tốc độ.
- Các ảnh hiện có trong `figures/` thuộc những run khác; không chèn chúng vào báo cáo này vì không khớp với cấu hình `lr=0.1` được phân tích. Cần xuất lại biểu đồ từ các run trong notebook để hoàn tất bộ hình nộp.
- Kết quả eval và phân tích lỗi ở mục 4 lấy từ output của `lab.ipynb`. Cần tạo `predictions_eval.csv` và `eval_result.json` từ đúng run `base-s2` (seed 2, SGD+momentum, `lr=0.1`) và chạy lại evaluator trước khi nộp.
- `experiments.xlsx` hiện đã có hai hàng đối chứng `loss-ce-s0` và `loss-mse-s0`; cần bổ sung các run khác được nêu trong report trước khi nộp toàn bộ hồ sơ. Các hình hiện có là hai đường loss riêng và hình so sánh CE-vs-MSE.
- Nếu có thêm thời gian, tôi sẽ lưu toàn bộ cấu hình sweep vào bảng kết quả, chạy thêm seed cho learning-rate/dropout comparison, thử class weighting cho lớp 4, và xuất lại dự đoán cùng evaluator JSON từ chính run cuối.

## 7. Phụ lục

- Đã tạo/cập nhật: `REPORT.md`, `experiments.xlsx` (hai hàng CE/MSE), `figures/compare_loss.png`, `figures/loss-ce-s0.png`, `figures/loss-mse-s0.png`, `results/loss-ce-s0.json`, `results/loss-mse-s0.json` và `code/`.
- Còn thiếu để đóng gói hoàn chỉnh: `predictions_eval.csv`, `eval_result.json`, các hàng bảng và biểu đồ cho những run khác được nêu trong report.
- Các run được nêu: `loss-ce-s0`, `loss-mse-s0`, `lrsweep-0.01`, `lrsweep-0.03`, `lrsweep-0.05`, `lrsweep-0.1`, `base-s0`, `base-s1`, `base-s2`, `dropout-0.3`.
- Tổng thời gian thực nghiệm không được ghi lại một cách đáng tin cậy; các giá trị timing trong summary không phải thời gian epoch riêng lẻ.
