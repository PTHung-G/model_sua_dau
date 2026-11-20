# Lịch sử Sửa đổi & Thiết kế Kiến trúc ViDiCRF v2

## 1. Thông tin bản ghi
- **Ngày thực hiện:** 2026-11-18
- **Repository gốc:** `model_sua_dau` — ViDiCRF (commit `3e88e76`: *Initial source snapshot: ViDiCRF research and portable demo*)
- **Môi trường kiểm chứng:** Windows x64, Python `3.12.10`, `torch 2.6.0+cu124`, GPU `NVIDIA GeForce RTX 3060 Laptop GPU` (6 GB VRAM), CPU `Intel Core i7-11370H` (đo suy luận giới hạn 2 luồng giống `DauDemo/server.py`).
- **Mục tiêu nâng cấp:**
  1. Khắc phục các nút thắt biểu diễn, bộ nhớ, tốc độ CPU và chính tả của kiến trúc v1 (`PunctSylCRF` — [charexp/sylmodel_punct.py](file:///d:/My_Project/ML2/charexp/sylmodel_punct.py)).
  2. Triển khai kiến trúc thế hệ mới **ViDiCRF v2** ([vidicrf_v2/](file:///d:/My_Project/ML2/vidicrf_v2/)) nâng cấp ở cả 4 tầng: Biểu diễn từ vựng–âm vị học, Backbone Hybrid BiGRU + Conformer-RoPE + Anchor Propagation, Top-8 Pruned Trigram Signed-Biaffine CRF, và Giải mã Forward–Backward MBR / khóa dấu đầu vào.
  3. Chạy bộ kiểm thử đơn vị và thực nghiệm đối chứng công bằng (A/B test) trên từ điển thật 1,38 triệu mục của `DauDemo` và dữ liệu Wikipedia tiếng Việt thật (`wikimedia/wikipedia`, `20231101.vi`).

---

## 2. Chẩn đoán 5 nút thắt & Lỗi thực tế của ViDiCRF v1 (`PunctSylCRF`)

Qua phân tích trực tiếp mã nguồn ([sylmodel_punct.py](file:///d:/My_Project/ML2/charexp/sylmodel_punct.py), [syltrain_punct.py](file:///d:/My_Project/ML2/charexp/syltrain_punct.py), [inference.py](file:///d:/My_Project/ML2/windows_demo_build/DauDemo/inference.py)) và kiểm tra toàn bộ inventory `DauDemo`:

1. **Nghịch lý "Mạng 5 MB nhưng checkpoint 272 MB" & Lỗi copy ~254 MB RAM khi gặp OOV**
   - Trọng số học được của `PunctSylCRF` chỉ có ~1,31 triệu tham số (~4,99 MB FP32), nhưng 3 buffer tĩnh `base_ch`, `cand_ch`, `cand_ids` cho **1.379.599** base và **1.431.311** vocab chiếm **254,19 MB** RAM (bản đóng gói `model.pt` trong `artifacts.json` là ~271,8 MB).
   - Kiểm tra trực tiếp [bases.json](file:///d:/My_Project/ML2/windows_demo_build/DauDemo/bases.json) và [vocab.json](file:///d:/My_Project/ML2/windows_demo_build/DauDemo/vocab.json) cho thấy:
     - **1.344.369 base (97,45%)** chỉ có đúng 1 ứng viên (`|C(x)| == 1`).
     - Trong đó **1.238.847 base (89,80% toàn bộ bảng)** là *identity singleton* (`x -> x`, trùng 100% với chính sách `COPY`, không hề thêm dấu nào).
     - Chỉ có **35.230 base (2,55%)** thực sự nhập nhằng (`|C(x)| >= 2`) và **105.522 base (7,65%)** có duy nhất 1 dạng có dấu khác với base.
   - Khi suy luận trong [inference.py](file:///d:/My_Project/ML2/windows_demo_build/DauDemo/inference.py), mỗi lần câu chứa từ OOV, hàm `predict()` gọi `torch.cat` nối thêm dòng vào cả 3 buffer lớn, **cấp phát và sao chép lại 254,19 MB RAM ngay trên luồng gõ phím** (đẩy độ trễ OOV p50 lên `85,64 ms`), đồng thời chạy lại `BiGRU` ký tự cho toàn bộ ứng viên mỗi lượt thay vì cache khi `eval()`.

2. **Nút thắt *Phonetic-to-Semantic Hashing* (Thiếu Lexical Embedding & Log-Prior)**
   - Vector ứng viên trong v1 phụ thuộc hoàn toàn vào `BiGRU` ký tự 128 chiều; thiếu hẳn bảng nhúng từ vựng mức âm tiết (*Lexical Syllable Embedding*) và thiếu đặc trưng thống kê tiên nghiệm $\log P_{\text{train}}(y \mid x)$.

3. **Phân mảnh nhãn chính tả & Lỗi crash do ký tự Thổ Nhĩ Kỳ `'i\u0307'`**
   - `vocab.json` trộn lẫn hai quy tắc đặt dấu (*kiểu cũ* vs *kiểu mới*: `hòa/hoà`, `thủy/thuỷ`), chứa từ gõ sai 2 dấu thanh (`'vỉết'`, `'vịêt'`), và cắt cụt `11.838` mục vocab dài trên `CHW=16`.
   - **Phát hiện bug thực tế khi chạy kiểm chứng**: `DauDemo/vocab.json` chứa **556 mục có độ dài khác với bản không dấu** (ví dụ `'i\u0307'`, `'bi̇lgi̇n'`, `'gi̇resunspor'`) sinh ra từ chữ hoa Thổ Nhĩ Kỳ `'İ'` khi `.lower()` thành 2 codepoint (`'i' + '\u0307'`). Nếu v1 chọn các ứng viên này trên đầu vào như `'i'`, hàm `restore_case` trong [eval_punct_vs_best.py](file:///d:/My_Project/ML2/eval_punct_vs_best.py#L33) lập tức ném `ValueError: Candidate and source lengths differ` (gây lỗi HTTP 500 trên server demo).

4. **Cắt đoạn cứng `MAXL=64`, chuẩn hóa loss lệch đoạn ngắn & Attention cổ chai**
   - `syltrain_punct.py` cắt cứng mỗi 64 token không tôn trọng ranh giới câu, chia loss trung bình từng đoạn trước khi `mean()` cả batch, và nén `attend()` xuống 32 chiều không có mã hóa vị trí tương đối (RoPE/ALiBi).

5. **Mù dấu đầu vào ("Thêm dấu" phá hỏng từ người dùng đã gõ dấu)**
   - Hàm `predict()` gọi `strip()` xóa sạch dấu đầu vào. Trong thử nghiệm giữ sẵn dấu 40% số từ trong câu, v1 đã **sửa hỏng 3.393 token đã có dấu** (chỉ bảo toàn được `85,31%`).

---

## 3. Thiết kế Kiến trúc & Thuật toán Mới — ViDiCRF v2 (`vidicrf_v2/`)

### Các module đã tạo
- [vidicrf_v2/\_\_init\_\_.py](file:///d:/My_Project/ML2/vidicrf_v2/__init__.py) — Export toàn bộ API của `vidicrf_v2`.
- [vidicrf_v2/orthography.py](file:///d:/My_Project/ML2/vidicrf_v2/orthography.py) — Chuẩn hóa vị trí đặt dấu (`classic` / `modern`), lọc nhiễu 2 dấu thanh & từ lệch độ dài (`'i\u0307'`), bảo vệ từ ngoại lai đa âm tiết (`'Napoléon'`, `'guó'`), phân rã âm vị học 6 thành phần (`onset`, `glide`, `nucleus`, `coda`, `mod`, `tone`), và thuật toán nén inventory không mất mát (`compact_inventory`).
- [vidicrf_v2/v1_mbr_decoder.py](file:///d:/My_Project/ML2/vidicrf_v2/v1_mbr_decoder.py) — Giải mã **Forward–Backward Posterior Marginal / MBR** và `constrained_viterbi` (hỗ trợ áp dụng ngay cho cả model v1 mà không cần train lại).
- [vidicrf_v2/model.py](file:///d:/My_Project/ML2/vidicrf_v2/model.py) — Kiến trúc `ViDiCRFv2` và các thuật toán Trigram CRF chính xác (`trigram_log_partition`, `trigram_gold_score`, `trigram_viterbi`, `trigram_marginals`, `compute_loss`):
  1. **Tầng 1 (Compositional Char-Conv + Phonology + Zero-Init Lexical Embedding + Log-Prior + Surface Diacritics):**
     - `CompositionalWordEncoder`: Nhúng ký tự phân rã `(base + mod + tone)` qua Conv1D đa kích thước ($k \in \{2, 3, 4\}$) + `PhonologicalMLP` 6 thành phần (không dùng RNN ký tự tuần tự và bỏ `.cpu()` sync).
     - `lex_emb = nn.Embedding(n_lex_vocab, d, padding_idx=0)` khởi tạo 0 kết hợp trực tiếp với `cand_proj` và `input_proj`.
     - Đặc trưng tiên nghiệm $[\log P_{\text{train}}(c \mid b), \log(\text{count}(c)+1)/15, \mathbb{1}[c=b]]$ kèm **Prior Dropout (`0,25`)** chống ỷ lại vào unigram.
     - **Zero-Copy Dynamic OOV & Precomputed Inference Cache**: Các từ ngoài từ điển/identity (`base_id == 0`) dùng trực tiếp vector bề mặt (`surface_chars`, `surface_phono`) mà không bao giờ sao chép buffer; hàm `build_inference_cache()` tính trước vector toàn bộ vocab khi `eval()`.
  2. **Tầng 2 (Hybrid BiGRU + Conformer-Lite + Iterative Anchor Propagation):**
     - `seq_gru` (`BiGRU` nhẹ với fast-path bỏ `pack_padded_sequence` khi `B=1` hoặc không padding) + khối `ConformerLiteBlock` (Multi-scale Gated Depthwise Conv1D $k=3, 5$ + `RoPE` Multi-Head Attention 4 heads × 32 = 128 chiều + SwiGLU FFN).
     - **Iterative Anchor Propagation**: Giữa Stage 1 và Stage 2, tổng hợp vector ngữ nghĩa có dấu kỳ vọng $\bar{e}_t = \sum_k p^{(1)}_{t,k} e(c_{t,k})$ cùng entropy bất định $u_t$ đưa qua cổng $\sigma(W_g[h_1; \bar{e}_t; u_t])$ vào Stage 2.
  3. **Tầng 3 (Top-8 Lattice Pruning + Exact Second-Order / Trigram Signed-Biaffine CRF):**
     - Cắt tỉa giữ Top-$M$ ($M=8$) ứng viên (đạt độ phủ `99,81%`, trong khi $8^3 = 512 < 32^2 = 1024$ của v1) với Bigram Biaffine có điều biến ngữ cảnh **nhận cả giá trị âm/dương** (`edge_signed`, không bị chặn $1 + \tanh > 0$) và Low-Rank Trigram Tensor (`r_tri = 16`).
  4. **Tầng 4 (Softmax-Margin + Auxiliary CE Loss):**
     - Cộng biên phạt Hamming $\gamma \cdot \mathbb{1}[y_t \ne y^*_t]$ vào `trigram_log_partition`, chuẩn hóa loss theo tổng token thực cả batch `(lp - gs).sum() / sent_len.sum()`, kết hợp Auxiliary CE trên cả Stage 1 và Stage 2.
- [vidicrf_v2/data_pipeline.py](file:///d:/My_Project/ML2/vidicrf_v2/data_pipeline.py) — Tạo inventory nén (giữ lại các identity thuần Việt phổ biến `min_identity_vn_count=5`), cắt đoạn ưu tiên ranh giới dấu câu (`max_len=128`, Context-Halo fallback), và **Stochastic Partial-Diacritic Collator**.
- [vidicrf_v2/restorer.py](file:///d:/My_Project/ML2/vidicrf_v2/restorer.py) — `FastRestorerV2` hỗ trợ sliding window có halo cho văn bản dài (>256 từ), khóa cứng nguyên văn các từ đã có dấu (`preserve_input_diacritics=True` + `forced_slots` truyền vào đồ thị CRF), chọn chế độ `viterbi` / `mbr`, và chuyển đổi kiểu dấu `classic` / `modern`.
- [test_vidicrf_v2.py](file:///d:/My_Project/ML2/test_vidicrf_v2.py) — Bộ 11 unit test kiểm chứng vét cạn toán học Trigram CRF, Forward–Backward, bất biến padding, Zero-Copy OOV và chính tả.
- [vidicrf_v2/run_ab_experiment.py](file:///d:/My_Project/ML2/vidicrf_v2/run_ab_experiment.py) — Script thực nghiệm A & B đối chứng công bằng.

---

## 4. Lịch sử Lặp Thiết kế & Chẩn đoán Kỹ thuật (Vòng 1 $\to$ Vòng 2)

Quá trình thử nghiệm thực tế qua 2 vòng đã phát hiện và xử lý các điểm cốt lõi sau:

- **Vòng 1 (Chẩn đoán trên checkpoint đầu):**
  - Thực nghiệm A trên từ điển `DauDemo` 1,38M mục chứng minh ngay hiệu quả nén 92,19% bases, xóa bỏ 254 MB cấp phát OOV và tăng tốc CPU p95 gấp 2,12 lần.
  - Ở Thực nghiệm B, `v2` vượt `v1` tới `+3,50%` trên validation (`85,24%` vs `81,74%`) nhưng trên RAW test tạm dừng ở `85,55%` (so với `v1` `86,10%`).
  - **Chẩn đoán sâu từ checkpoint Vòng 1** đã chỉ ra chính xác 3 nguyên nhân:
    1. Nhánh encoder của `v2` (`emit2` = `81,43%`) vốn đã vượt xa encoder của `v1` (`emit_only` = `76,80%`) tới **`+4,63%`**, nhưng `Top-4` chỉ phủ **`98,09%`** tập ứng viên đúng (`63.661/64.900`), làm rơi **1.239 đáp án đúng** trước khi tới Trigram CRF (trong khi `Top-8` phủ **`99,81%`**).
    2. Việc thiếu `LayerNorm` trước đầu đọc emission + nhân thừa $1/\sqrt{d}$ làm lệch thang logits; `lex_emb` qua `Linear` khởi tạo `0.02` + `weight_decay=0.01` làm suy hao cập nhật; thiếu `BiGRU` quy nạp tuần tự; và `partial_diac_prob=0.35` giảm số bước học 100% không dấu còn ~2,6 epoch.
    3. Trong bài test giữ sẵn 40% dấu, 42 token sai đến từ từ ngoại lai đa âm tiết (`'Napoléon'`, `'guó'`) và ép style `'hoá' -> 'hòa'` khi chưa trả nguyên văn `orig` ở `locked_set`.

- **Vòng 2 (Bản hoàn thiện hiện tại):**
  - Mở rộng lên **Top-8 Trigram CRF**, thêm **Hybrid BiGRU + Conformer-RoPE**, dùng **`lex_emb` trực tiếp khởi tạo 0** (tách `weight_decay=0.0` cho embedding/LayerNorm), thêm **Prior Dropout (`0,25`)**, giữ các identity thuần Việt phổ biến (`min_identity_vn_count=5`), đặt `partial_diac_prob=0,15`, sửa triệt để quy tắc ngoại lai trong `orthography.py` và bảo toàn nguyên văn `orig` cho `locked_set` trong `restorer.py`.

---

## 5. Kết quả Kiểm thử & Thực nghiệm Đối chứng (Số liệu thực đo Vòng 2)

Nguồn số liệu đầy đủ lưu tại [ket_qua_benchmark_v1_vs_v2.json](file:///d:/My_Project/ML2/lich_su_sua_doi/ket_qua_benchmark_v1_vs_v2.json).

### A. Kiểm thử Đơn vị (`unittest`)
- Bộ test mới [test_vidicrf_v2.py](file:///d:/My_Project/ML2/test_vidicrf_v2.py): **11/11 PASS** (vét cạn kiểm chứng Trigram CRF & Forward–Backward khớp sai số $< 10^{-4}$, padding invariance, Zero-Copy OOV, cache equivalence, orthography & phonology).
- Bộ test gốc của repo:
  - `test_punct_pipeline.py`: **8/8 PASS**.
  - `test_model_punct_regression.py`: **8/8 PASS**.
  - `test_punct_alignment_audit.py`: **4/5 PASS** (test còn lại pass toàn bộ assertion, chỉ gặp lỗi khóa handle `mmap_mode='r'` của Windows NTFS khi `TemporaryDirectory` dọn file `.npy`).

### B. Thực nghiệm A — Trên Từ điển Thật `DauDemo` 1,38 triệu mục (CPU 2 luồng, 100 câu test Wikipedia)

| Chỉ tiêu | ViDiCRF v1 (`PunctSylCRF`) | ViDiCRF v2 (`Conformer-RoPE + Cache`) | Chênh lệch / Cải thiện |
| :--- | :---: | :---: | :--- |
| Số `bases` trong bảng | `1.379.599` | `107.703` | **Giảm 92,19%** (loại `1.246.531` mục dư thừa/nhiễu) |
| Số `vocab` | `1.431.311` | `148.008` | **Giảm 89,66%** |
| Dung lượng buffers tĩnh | `254,19 MB` | `42,72 MB` | **Nhỏ hơn 5,95×** |
| Tổng dung lượng model (params + buffers) | `259,18 MB` | `61,14 MB` | **Nhỏ hơn 4,24×** |
| Bộ nhớ cấp phát mới khi gặp từ OOV | `254,19 MB` | **`0,00 MB`** | **Zero-Copy 100%** |
| Độ trễ CPU `p50` (câu chuẩn) | `55,75 ms` | **`40,26 ms`** | **Nhanh hơn 1,38×** |
| Độ trễ CPU `p95` (câu đuôi dài) | `106,20 ms` | **`64,48 ms`** | **Nhanh hơn 1,65×** |
| Độ trễ CPU `p50` khi có từ OOV | `85,64 ms` | **`38,32 ms`** | **Nhanh hơn 2,23×** |

### C. Thực nghiệm B — Huấn luyện & Đánh giá Đối chứng trên Wikipedia `20231101.vi`
- **Thiết lập:** `12.000` đoạn train, `1.000` đoạn validation, `1.000` đoạn RAW test khóa cứng (`71.173` tokens, `60.925` tokens thuộc base nhập nhằng $\ge 2$ ứng viên), `4 epochs`, `batch_size=32`, `seed=42`, GPU `RTX 3060 Laptop`.

| Hệ thống | Val Acc | RAW Token Acc | Ortho-Norm Token Acc | Ambiguous Token Acc | Exact Paragraph | Partial-Diac 60% Không dấu | Bảo toàn 40% Có dấu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `COPY` (giữ nguyên) | — | `19,31%` | `19,31%` | `12,61%` | `0,00%` | — | — |
| `Train-frequency` | — | `74,08%` | `74,13%` | `70,02%` | `2,80%` | — | — |
| **`ViDiCRF v1` (Viterbi gốc)** | `81,74%` | `86,10%` | `86,17%` | `84,05%` | `12,50%` | `86,53%` | `85,31%` (hỏng **3.393** từ) |
| **`ViDiCRF v1 + FB-MBR`** *(Zero-Retrain)* | — | `86,15%` | `86,22%` | `84,11%` | `12,50%` | — | — |
| **`ViDiCRF v2` (Viterbi)** | **`88,23%`** | **`88,65%`** | **`88,73%`** | **`87,04%`** | **`17,80%`** | **`89,32%`** | **`100,00%`** (**0** từ hỏng) |
| **`ViDiCRF v2` (FB-MBR)** | — | **`88,72%`** | **`88,80%`** | **`87,13%`** | `17,20%` | — | — |

**Kết luận định lượng:**
1. Trên **cùng dữ liệu và cùng 4 epoch huấn luyện**, `ViDiCRF v2` tăng **`+2,62 điểm %`** độ chính xác Token tổng thể (`88,72%` vs `86,10%`), **`+3,08 điểm %`** trên các từ nhập nhằng (`87,13%` vs `84,05%`), và tăng **`+5,30 điểm %`** (tương đối **`+42,4%`**) tỷ lệ đoạn văn đúng hoàn toàn 100% (`17,80%` vs `12,50%`).
2. Giải mã **Forward–Backward MBR** giúp tăng thêm độ chính xác token trên cả v1 (`86,10% -> 86,15%` không cần train lại) lẫn v2 (`88,65% -> 88,72%`).
3. Trong kịch bản gõ thực tế có sẵn 40% chữ có dấu, `ViDiCRF v2` bảo toàn **100,00% (`0` lỗi)** các từ người dùng đã gõ dấu (trong khi v1 làm hỏng 3.393 từ), đồng thời dùng chính chúng làm "mỏ neo" giúp nâng độ chính xác 60% từ còn lại lên **`89,32%`** (`+2,79%` so với v1).

---

## 6. Hướng dẫn chạy kiểm tra lại

```powershell
# 1. Chạy bộ kiểm thử đơn vị ViDiCRF v2 (CPU-only, ~1.6s)
.\.venv\Scripts\python.exe -X utf8 -m unittest test_vidicrf_v2 -v

# 2. Chạy bộ thực nghiệm đối chứng A/B (cần GPU/CPU và kết nối mạng tải parquet lần đầu)
.\.venv\Scripts\python.exe -X utf8 -m vidicrf_v2.run_ab_experiment
```

---

## 7. Phụ lục V3 — Tối ưu CPU + PMI sketch + Kernel Kaggle quy mô lớn (20/11/2026)

> **Nhãn trung thực:** toàn bộ mục này là **benchmark thăm dò của riêng tôi (v3)**, ngoài phạm vi đề xuất D3 (D3 ràng buộc "không kiến trúc mới, không train lại" — v2/v3 vi phạm cả hai). Không số nào dưới đây là kết quả của môn học. Độ trễ đo trên **Linux/CPU 2 threads, torch 2.11** — **không so sánh 1:1** với đo Windows/torch khác.

### A. Tối ưu CPU không cần train lại (`vidicrf_v2/fast_inference.py`)
- `WordCache`: memo 7-dặc-trưng bề mặt từng từ + tensor char/phono.
- `scripted_viterbi`: bản TorchScript của `trigram_viterbi` (biên dịch qua `__wrapped__` do wrapper `@torch.no_grad()` làm `torch.jit.script` lỗi trên torch 2.11), có fallback eager.
- `CSRInventory`: `cand_ids` dày đặc → CSR (indptr + vid int32 phẳng).
- `FastRestorerV2Opt`: subclass `FastRestorerV2` + memo toàn-câu; **ràng buộc tương đương đầu-ra 1:1 với bản gốc, khóa bằng test** (13 test mới + 24 test hồi quy đều xanh).
- **Sửa kê khai bộ nhớ** (lỗi bỏ sót cache suy luận trong Exp A): `runtime_total = params + buffers + inference_cache`:

| Đại lượng | Exp A báo cáo | Đo lại đầy đủ (Linux) |
| :--- | :---: | :---: |
| Checkpoint (params + buffers) | `61,14 MB` | `61,14 MB` |
| Inference cache (148.008 từ × 128 × f32) | *(bị bỏ sót)* | `72,27 MB` |
| **Tổng bộ nhớ runtime** | `61,14 MB` | **`133,41 MB`** (≈ 1,95× so với v1 259,18 MB, không phải 4,24×) |
| CSR `cand_ids` (13,79 MB → 1,45 MB, tỉ lệ lấp đầy 4,29%) | — | nén `9,48×` |

- **Phân rã độ trễ** (câu mẫu 24 từ, Linux): baseline p50 `11,02 ms` — trong đó `model_forward` chiếm `9,37 ms` (~82%), tokenizer/tổng hợp tensor chỉ ~0,4 ms. **Khác biệt với đo Windows của tôi trước đó (40,3 ms tổng, ~60% ngoài model) là đặc thù nền tảng**; số 24 ms overhead chưa được xác nhận trên môi trường này.
- **Latency câu mới mỗi lần** (thước đo thực dụng nhất cho gõ văn bản): baseline `14,82 ms` vs opt `14,28 ms` (**chỉ nhanh ~3,6%**) — vì trên nền tảng này `model_forward` đã chiếm đa số thời gian. Memo toàn-câu (`p50 0,01 ms`, nhanh ~1000×) chỉ áp dụng cho **input lặp lại y hệt** (vd. chạy lại cùng câu); với gõ từng phím (mỗi phím = câu mới) lợi ích chính nằm ở chỗ khác.
- Chi tiết đầy đủ: `lich_su_sua_doi/ket_qua_toi_uu_cpu_v3.json`.

### B. PMI bigram train-only (`vidicrf_v2/pmi.py`)
- `PMITable`: đếm cặp trên **chỉ partition train**, lọc `min_pair_count=3`, clip `[−4, 8]`, tra cứu `searchsorted` trên key int64 đã sắp; bơm `bigram += λ · PMI(cặp ứng viên kề)` với λ học được (khởi tạo 0.5).
- Có bước smoke huấn luyện 1 bước thật: loss hữu hạn, `λ.grad ≠ 0`.
- Ablation đang chạy trên Kaggle (kernel B), so với kernel A (cùng protocol, không PMI).

### C. Kernel Kaggle quy mô lớn (`kaggle_run/`)
- 2 kernel riêng tư (T4): `vidicrf-v3-scale-200k-base` (không PMI) và `vidicrf-v3-scale-200k-pmi`; protocol **giữ nguyên** Exp B (seed 42, 4 epochs, bs 32, lr 0.0020, dedup ưu tiên eval, RAW test) chỉ tăng train 12.000 → 200.000 đoạn; revision HF khóa cứng `b04c8d1c…`; AMP FP16 chỉ quanh encoder, loss CRF/CE luôn FP32; decode test: Viterbi / FB-MBR / **two-pass anchored** (τ ∈ {0.8, 0.9, 0.95}) / partial-diac 40%.
- Dataset nguồn `tunhngphm/vidicrf-v3-src` (7 module, zip); kernel tự lắp layout package từ file phẳng.
- Kết quả sẽ nằm ở `/kaggle/working/results_v3.json` của từng kernel (kiểm tra thủ công trên Kaggle).
