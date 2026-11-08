# DS201 - Lab 3: Mạng neural hồi quy (LSTM, GRU, BiLSTM)

Repo gồm mã nguồn và kết quả cho 4 bài thuộc Phần C của Lab 3. Báo cáo
tổng hợp cả 4 bài nằm trong `report.pdf`.

## Các bài

| Bài | Nhiệm vụ | Mô hình | Dữ liệu | Script |
|---|---|---|---|---|
| 1 | Phân loại cảm xúc | LSTM 5 lớp | UIT-VSFC | `bai1_lstm_sentiment.py` |
| 2 | Phân loại cảm xúc | GRU 5 lớp, so sánh với Bài 1 | UIT-VSFC | `bai2_gru_sentiment.py` |
| 3 | Nhận diện thực thể (NER) | BiLSTM 5 lớp | PhoNER_COVID19 (syllable) | `bai3_bilstm_ner.py` |
| 4 | So sánh độ sâu | LSTM với 1, 2, 5 lớp | UIT-VSFC | `bai4_depth.py` |

Cấu hình chung: embedding 300, hidden 256, dropout 0.3, batch 32,
Adam (lr 1e-3), clip gradient 1.0, seed 42, tối đa 10 epoch. Embedding
học từ đầu, không dùng mô hình pretrained. Checkpoint được chọn theo
F1 trên tập dev, sau đó mới đánh giá trên test.

## Cấu trúc thư mục

```
data.py                  tải và kiểm tra dữ liệu, từ điển, DataLoader
models.py                bộ phân loại câu (RNN/LSTM/GRU) và BiLSTM tagger
engine.py                cấu hình, huấn luyện, độ đo, baseline TF-IDF, xuất kết quả
bai1_lstm_sentiment.py
bai2_gru_sentiment.py
bai3_bilstm_ner.py
bai4_depth.py
report.pdf               báo cáo
results/                 số liệu và hình của từng bài (có trong repo)
data/                    tự tạo khi chạy, không có trong repo
checkpoints/             tự tạo khi chạy, không có trong repo
```

## Môi trường

- Python 3.10 trở lên
- Cài thư viện: `pip install -r requirements.txt`
- Nên chạy bằng GPU, chạy trên CPU rất chậm.
- Môi trường đã dùng để chạy ra kết quả trong báo cáo: [ghi loại GPU,
  phiên bản Python, PyTorch]. Thông tin này cũng được lưu trong
  `results/*/results.json`.

## Dữ liệu

Dữ liệu không nằm trong repo. Lần chạy đầu tiên, `data.py` tự tải về
thư mục `data/` và kiểm tra SHA-256. Các lần sau dùng lại tệp đã có.

- UIT-VSFC (Vietnamese Students' Feedback Corpus): chỉ dùng nhãn
  sentiment (negative, neutral, positive).
  Số mẫu: 11.426 train, 1.583 dev, 3.166 test.
- PhoNER_COVID19, bản syllable, từ kho VinAI:
  https://github.com/VinAIResearch/PhoNER_COVID19
  Số mẫu: 5.027 train, 2.000 dev, 3.000 test; 21 nhãn BIO.

Nếu tải tự động thất bại (link bị chặn hoặc trả về trang HTML), chương
trình dừng và báo tệp nào bị lỗi. Lúc này cần tự tải dữ liệu từ nguồn gốc và đặt
đúng tên như bên dưới:

```
data/uit-vsfc/UIT-VSFC-train.json
data/uit-vsfc/UIT-VSFC-dev.json
data/uit-vsfc/UIT-VSFC-test.json
data/phoner/train_syllable.json
data/phoner/dev_syllable.json
data/phoner/test_syllable.json
```

Danh sách link và checksum nằm ở biến `SOURCES` trong `data.py`. Tệp
đặt tay phải khớp SHA-256 thì chương trình mới chấp nhận.

## Cách chạy

Chạy từ thư mục gốc của repo:

```
pip install -r requirements.txt
python bai1_lstm.py
python bai2_gru.py     # cần kết quả của Bài 1
python bai3_bilstm.py
python bai4_depth.py
```

- Bài 2 đọc lại kết quả LSTM và TF-IDF từ `results/bai1/results.json`
  để so sánh, nên phải chạy Bài 1 trước.
- Bài 4 lưu kết quả sau từng cấu hình; nếu bị ngắt giữa chừng, chạy lại
  sẽ chỉ làm tiếp phần còn thiếu.

## Kết quả

Mỗi bài có một thư mục trong `results/`:

| Thư mục | Nội dung |
|---|---|
| `results/bai1/` | `results.json`, `test_metrics.csv`, `cost.csv`, `curves.png`, `confusion_dev.png`, `confusion_test.png` |
| `results/bai2/` | như Bài 1, kèm so sánh LSTM, GRU và TF-IDF |
| `results/bai3/` | `results.json`, `test_per_type.csv`, `curves.png` |
| `results/bai4/` | một tệp JSON cho mỗi (số lớp, seed), `depth_table_seed*.csv`, `curves_seed*.png` |

Các con số trong `report.pdf` lấy từ các tệp này.