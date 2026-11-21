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

### C. Kernel Kaggle (`kaggle_run/`) — 4 kernel, lưới 2×2
- 4 kernel riêng tư T4: `-200k-base`, `-200k-pmi`, `-12k-base`, `-12k-pmi`. Protocol **giữ nguyên** Exp B (seed 42, 4 epochs, bs 32, lr 0.0020, dedup ưu tiên eval, RAW test), **chỉ đổi `N_TRAIN`** (12.000 vs 200.000) và `VARIANT` (base vs pmi); revision HF khóa cứng `b04c8d1c…`; AMP chỉ quanh encoder, loss CRF/CE luôn FP32; decode test: Viterbi / FB-MBR / **two-pass anchored** (τ ∈ {0.8, 0.9, 0.95}) / partial-diac 40%.
- Dataset nguồn `tunhngphm/vidicrf-v3-src` (7 module, zip); kernel tự lắp layout package từ file phẳng.
- Kết quả: `lich_su_sua_doi/kaggle_v3/results_{base,pmi}_{12k,200k}.json`.

### D. Lưới kết quả 2×2 trên Kaggle T4 (20–21/11/2026) — benchmark thăm dò, ngoài D3
- Cả 4 lần chạy dùng **cùng một test set** 71.173 token (salt hash + pinned revision) → so sánh trực tiếp được, chỉ khác lượng train và việc bật/tắt PMI.
- 200k: train 194.495/200.000 đoạn (5.505 skip), 217.938 segments, inventory 25.303 dạng từ, ~5.880s/epoch, tổng ~5h51m train. 12k: ~13k segments, ~265–288s/epoch, ~18–19 phút train.
- AMP thực tế dùng **BF16** (torch 2.11 báo bf16 khả dụng trên T4 qua emulation; GradScaler tự tắt), CRF/CE luôn FP32; `nan_skips = 0` ở **cả 4 lần chạy**.
- **Kiểm tra độ trung thực**: bản tái lập 12k-base cho 88,72% so với 88,65% của Exp B gốc đã commit → lệch **+0,07 điểm**, tức vòng train trong kernel tái tạo đúng semantics của `run_ab_experiment.py` (chênh lệch nằm trong nhiễu do AMP). Đây là cơ sở để tin số 200k không bị méo bởi AMP.

| Hệ thống (cùng test set 71.173 token) | Val | **RAW Token** | Nhập nhằng | Exact Paragraph | Ortho-Norm |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 12k base (Viterbi) | 88,14% | 88,72% | 87,13% | 17,60% | 88,80% |
| 12k base (FB-MBR) | — | 88,84% | 87,26% | 17,50% | 88,92% |
| **12k PMI (Viterbi)** | **90,76%** | **91,29%** | **90,13%** | **22,00%** | **91,37%** |
| 12k PMI (FB-MBR) | — | 91,26% | 90,09% | 21,40% | 91,34% |
| 200k base (Viterbi) | 94,65% | 96,34% | 96,09% | 40,30% | 96,44% |
| 200k base (FB-MBR) | — | 96,35% | 96,11% | 40,00% | 96,46% |
| **200k PMI (Viterbi)** | **94,87%** | **96,60%** | **96,38%** | **41,60%** | **96,70%** |
| 200k PMI (FB-MBR) | — | 96,60% | 96,38% | 41,40% | 96,70% |

### E. Kết luận chính: PMI là công cụ cho chế độ **ít dữ liệu**, không phải cho quy mô lớn
- **Lợi ích PMI co lại ~10 lần khi tăng dữ liệu**:

| Chỉ số | 12k (PMI − base) | 200k (PMI − base) | Tỉ lệ |
| :--- | :---: | :---: | :---: |
| RAW token | **+2,57** | +0,26 | 9,9× |
| Nhập nhằng | **+3,00** | +0,29 | 10,3× |
| Exact paragraph | **+4,40** (+25,0% tương đối) | +1,30 (+3,2%) | 3,4× |
| Ortho-Norm | **+2,57** | +0,26 | 9,9× |
| Val acc | **+2,62** | +0,22 | 11,9× |

- Nhìn theo **giảm lỗi tương đối** (thang đo đúng cho ablation): 12k giảm **22,8% lỗi** token (11,28% → 8,71%); 200k chỉ giảm **7,1%** (3,66% → 3,40%). Ở 200k vẫn có lãi, nhưng phần lớn thống kê bigram đã được model neural tự nội hóa từ 194k đoạn → sketch ngoài chỉ còn phần dư nhỏ.
- **λ học được KHÔNG phải chỉ báo độ hữu ích**: hội tụ về **1,0990 ở 12k** và **1,0965 ở 200k** — gần như trùng nhau, dù lợi ích chênh 10 lần. Nghĩa là model luôn "muốn" prior này ở trọng số ~1,1; khác biệt nằm ở **lượng thông tin dư** trong tín hiệu, không nằm ở scalar. (Đính chính lại cách đọc λ ở mục D bản trước: λ cao không chứng minh PMI có ích.)
- **Quy đổi tương đương dữ liệu**: nội suy log-tuyến tính qua 2 điểm (12k: 88,72%; 200k: 96,34% — độ dốc 2,71 điểm/ln) cho thấy +2,57 điểm của PMI ở 12k tương đương **≲2,6× dữ liệu**. Đây là **cận trên**, vì đường cong học là lõm (lợi ích giảm dần) nên độ dốc cục bộ tại 12k cao hơn độ dốc trung bình → quy đổi thật nhỏ hơn. Không thể thay dữ liệu bằng PMI: 12k+PMI (91,29%) vẫn kém 200k base (96,34%) **5,05 điểm**.
- Bảng PMI ở 12k chỉ có 36.014 cặp (phủ 2.847/25.303 từ) so với 371.018 cặp ở 200k (phủ 11.866 từ) — chính vì thế mà ở 12k sketch vừa *hữu ích hơn* (bù dữ liệu thiếu) vừa *mỏng hơn* (ít cặp học được).
- **Two-pass anchored phẳng tuyệt đối ở cả 4 lần chạy**: 88,72 / 91,29 / 96,34 / 96,60 giống hệt nhau tại τ ∈ {0.8, 0.9, 0.95} — không đổi một prediction nào. Kết quả tiêu cực rất chắc (4 cấu hình × 3 ngưỡng = 12 phép thử): anchor-cứng trên cùng lattice không tạo ra thông tin mới, vì MBR và Viterbi gần như luôn đồng ý → **tiền đề cho "2-step BP / candidate cross-attention" đầy đủ yếu, nên hạ ưu tiên**.
- **Bảo toàn dấu người dùng tăng mạnh theo dữ liệu** (free-decode, không hard-copy): 12k base bảo toàn 88,22% (2.720 từ hỏng) → 200k base **99,66%** (78 từ hỏng). Với ít dữ liệu model "sửa quá tay" cả những từ người dùng đã gõ đúng dấu. PMI giảm số từ hỏng **~24–30% ở cả hai mức** (2.720 → 2.081; 78 → 55) — hiệu ứng ổn định và nhất quán nhất của PMI.
- Chi phí thời gian PMI: ~0% ở 200k (5.264 vs 5.265s/epoch), +8,7% ở 12k (288 vs 265s/epoch) — chênh lệch nhỏ, nằm trong nhiễu đo/dao động xung nhịp GPU, không phải xu hướng thật.

### F. Lưu ý protocol & chấm điểm dự đoán
- **Partial-diac ở 4 lần chạy Kaggle dùng protocol khác A/B 12k**: decode **tự do**, không hard-copy từ có dấu sẵn như restorer (A/B khóa cứng → 100% bảo toàn). Vì vậy **chỉ so sánh chéo trong lưới 2×2**, không so với cột 89,32%/100% của bảng Exp B.
- Phần trăm bảo toàn dấu trong lưới: 12k 88,22% → 90,99% (PMI); 200k 99,66% → 99,76% (PMI). Accuracy 60%-không-dấu: 89,20 → 91,67 (12k); 96,61 → 96,88 (200k).

**Chấm điểm dự đoán 200k** (ghi trước khi chạy):
| Dự đoán | Thực tế | Kết quả |
| :--- | :--- | :---: |
| base test ~93,5–95% | 96,34% | ✗ (thấp quá) |
| PMI hơn base +0,1–0,6 | +0,26 | ✓ |
| MBR hơn Viterbi +0,05–0,15 | +0,01 / +0,00 | ~ (đúng chiều "micro") |
| Delta nhập nhằng > delta tổng | +0,29 vs +0,26 | ✓ |
| Exact paragraph 28–40% | 40,3% / 41,6% | ✓ (sát mép trên) |
| Partial 60% không dấu ~94–95,5% | 96,61% / 96,88% | ✗ (thấp quá) |
| λ cuối ~1,0–1,3 | 1,0965 | ✓ |
| Best checkpoint = ep4; nan=0 | đúng | ✓ |
| Two-pass phẳng (±0,15) | phẳng tuyệt đối 0,00 | ✓ |

**Chấm điểm dự đoán 12k** (ghi trước khi chạy):
| Dự đoán | Thực tế | Kết quả |
| :--- | :--- | :---: |
| Gap PMI ở 12k **lớn hơn** +0,26 của 200k | +2,57 (≈10×) | ✓ |
| Bản tái lập 12k-base ≈ 88,65% đã commit | 88,72% (lệch +0,07) | ✓ |
| Two-pass vẫn phẳng | phẳng tuyệt đối 0,00 | ✓ |
