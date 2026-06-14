# Tích hợp ý tưởng DyG-RAG & IA-RAG vào HippoRAG — Phân tích khả thi

> Artifact phục vụ báo cáo kỹ thuật (Expected Output #5). Soạn 2026-06-14.
> Nguồn: DyG-RAG (arXiv 2507.13396, code github.com/RingBDStack/DyG-RAG),
> IA-RAG (arXiv 2606.06044), khảo sát "It's High Time" (arXiv 2505.20243).
> Map code dựa trên `src/hipporag/HippoRAG.py` và cộng sự (file:line bên dưới).

---

## 0. Điểm mấu chốt: lệch kiến trúc nền tảng

| | HippoRAG | DyG-RAG / IA-RAG |
|---|---|---|
| Đơn vị nguyên tử | **Triple** (subject, predicate, object) | **Event** (1 câu = 1 sự kiện) có mốc/khoảng thời gian |
| Nút đồ thị | entity / passage / fact | DEU / IEU (event) |
| Cạnh | đồng-xuất-hiện entity + synonymy KNN | entity-overlap + temporal proximity (DyG) / Allen interval relations (IA) |
| Truy hồi | **PPR** (random walk trên đồ thị entity) | weighted random walk theo thời gian (DyG) / interval-algebra-guided traversal (IA) |
| Thời gian | scalar timestamp trên *cạnh* (hiện = 0 trên dữ liệu thật) | first-class: gắn vào *đơn vị event*, encode vào embedding, dùng để duyệt |

**Hệ quả:** KHÔNG bê nguyên cả hệ thống. Port toàn bộ Allen-graph + Thematic Forest = xây lại IA-RAG, HippoRAG chỉ còn là scaffolding OpenIE. Cách đúng: tách thành **ý tưởng module**, chỉ ghép cái nào *augment* được nền PPR-entity sẵn có mà không thay nền. Đó là tiêu chí chấm điểm bên dưới.

**Headroom:** paper báo HippoRAG baseline TimeQA ~40% / TempReason ~70% / ComplexTR ~45% (Accuracy); DyG/IA vượt +10…+22pp. Lý do chính: HippoRAG hiện **không time-aware** trên dữ liệu này (`timestamp=0` → `temporal_weighting` trơ).

---

## 1. Bóc tách ý tưởng + chấm điểm tương thích

Thang: Fit (mức ghép vào nền PPR-entity), Effort, Gain, Risk.

### Từ DyG-RAG

| # | Ý tưởng | Fit | Effort | Gain | Seam trong code |
|---|---|---|---|---|---|
| D1 | Per-fact time anchor — LLM trích mốc thời gian *cho từng triple* từ chunk (thay doc-level=0) | ★★★★★ | TB | Cao | `triple_extraction.py` prompt + `TripleRawOutput` (misc_utils) → `add_fact_edges` `_update_edge_meta` (HippoRAG.py:800-818) |
| D2 | Timeline + Time-CoT — sắp xếp evidence top-k theo thời gian, prompt QA có template suy luận thời gian | ★★★★★ | Thấp | Cao | Post-retrieval trong `retrieve()`/`qa()` + template `rag_qa_timeqa` mới |
| D3 | Query temporal intent → time-scoped retrieval — parse `question_time` (ĐÃ có sẵn) để thiên vị/lọc theo thời gian | ★★★★☆ | TB | Cao (trúng `time_specific`) | `get_fact_scores`/`dense_passage_retrieval` (HippoRAG.py:1347, 1387) |
| D4 | Fourier time encoding ghép vào embedding `z=[h_text; h_time]` | ★★★☆☆ | TB-Cao | TB | EmbeddingStore + query embedding (đổi chiều vector 2 phía) |
| D5 | Temporal-decay edge weight `w=sim·exp(−α·Δt)` cho PPR | ★★★★☆ | Thấp | TB | `_get_ppr_edge_weights()` (HippoRAG.py:1629) — đúng seam recency đang có |

### Từ IA-RAG

| # | Ý tưởng | Fit | Effort | Gain | Ghi chú |
|---|---|---|---|---|---|
| I1 | Interval `[t_start, t_end]` + fuzzy flag thay timestamp điểm | ★★★★☆ | TB | Cao | Khớp Goal 2 ("validity period") + cần cho `valid_from/valid_to`; sửa scalar→tuple trong plumbing |
| I2 | Query scope + directional traversal (Ψ before/after) — chỉ mở rộng neighbor theo hướng thời gian nhất quán | ★★★☆☆ | TB | Cao (multihop temporal) | Làm ở seed-expansion *ngoài* PPR; cần I1 |
| I3 | Allen Interval Algebra edges (13 quan hệ) như lớp cạnh mới | ★★☆☆☆ | Cao | TB | PPR chỉ nhận weight vô hướng, không hiểu *kiểu* quan hệ → dùng Allen ở query-time, không nhét vào PPR |
| I4 | Sub-graph Time Tightening — suy interval mờ từ neighbor explicit (LLM) | ★★★☆☆ | Cao | TB | Hấp dẫn cho conflict/supersede (Goal 3/4); nặng, phụ thuộc I1+I3 |
| I5 | Thematic Forest (MEU phân tầng, summarize đệ quy) | ★☆☆☆☆ | Rất cao | Thấp-TB | Lớp community-summary kiểu GraphRAG, đi ngược triết lý HippoRAG. Không khuyến nghị |
| I6 | IEU dedup bằng semantic-neighborhood + LLM | ★★☆☆☆ | Thấp | Thấp | HippoRAG đã có synonymy KNN + entity dedup → trùng lặp |

---

## 2. Cái KHÔNG nên port

- **I5 Thematic Forest**: tốn nhiều LLM-call summarize đệ quy (depth=4). HippoRAG cố tình không làm community summary (GraphRAG chậm vì cái này). Bất khả thi với 3B + 2×T4, lệch triết lý.
- **I3 Allen-graph nhét vào PPR**: `personalized_pagerank` của igraph (HippoRAG.py:1706) chỉ nhận `weights` vô hướng — không biểu diễn 13 kiểu quan hệ. Muốn dùng Allen phải duyệt ngoài PPR → bỏ nền HippoRAG. Chỉ mượn Allen ở query-time (I2) làm bộ lọc hướng.
- **D4 Fourier encoding** (giai đoạn đầu): đổi chiều embedding cả index lẫn query, phải re-embed toàn corpus. Có xấp xỉ rẻ hơn (D3: nhân prior thời gian vào fact score).

---

## 3. Đường thử nghiệm đề xuất (3 tầng)

Nguyên tắc: mỗi tầng độc lập đo được, không phá static benchmark (gate bằng config, mặc định off).

### Tầng 1 — Rẻ, ROI cao nhất (làm trước)
**D2 (Timeline + Time-CoT) + D1 (per-fact time anchor)**, đo trên ComplexTR (329 Q).
- D1 cấp mốc thời gian thật cho fact (hiện 0) → D2 mới sắp được timeline.
- D2 thuần post-processing + 1 template prompt mới → không đụng đồ thị/PPR, model-agnostic, dễ rollback.
- Đo: bật/tắt qua config, so Accuracy/Recall bằng `eval_dyg.py`.

### Tầng 2 — Time-scoped retrieval
**D3 + D5** (dùng `question_time` đã parse sẵn).
- D3: thêm prior thời gian lên fact score = xấp xỉ rẻ của D4 (không Fourier): `score *= proximity(question_time, fact_time)`. Sửa trong `get_fact_scores` (HippoRAG.py:1347).
- D5: đổi recency-PPR tuyến tính hiện tại sang decay/proximity quanh `question_time` trong `_get_ppr_edge_weights()` (HippoRAG.py:1629).
- Đo: trúng `time_specific` — đúng failure mode đã ghi nhận trước đây.

### Tầng 3 — Interval & conflict (gắn Goal 2/3/4)
**I1 (interval+fuzzy) → I2 (directional traversal) → I4 (time tightening) → conflict/supersede.**
- I1 nâng metadata scalar→interval (validity period) — Goal 2 còn dang dở.
- I4 (suy interval mờ từ neighbor) chính là cơ chế reasoning giữa các fact lân cận mà conflict-detection (Goal 3) cần — một mũi tên hai đích.
- Nặng, để sau khi Tầng 1-2 cho tín hiệu dương.

---

## 4. Cảnh báo điều kiện thí nghiệm

1. **LLM 3B-AWQ quá yếu cho D1, D2, I4** (đều dựa vào LLM trích/suy luận thời gian). Validate ý tưởng nên chạy ComplexTR nhỏ với LLM mạnh hơn (gpt-4o-mini API hoặc ≥14B như IA-RAG dùng Qwen2.5-14B). 3B chỉ để smoke-test pipeline.
2. **Tính so sánh**: giữ protocol DyG/IA (chunk 1200/64, metric Accuracy/Recall). Mọi cải tiến đo trên nền baseline HippoRAG đã reproduce, không phải nền tự chế.
3. **Phải có baseline trước**: chưa có số HippoRAG gốc trên timeqa/tempreason/complextr thì chưa cài cải tiến — cần điểm xuất phát để biết +bao nhiêu pp.

---

## 5. Kết luận

- **Tương thích cao, thử ngay**: D1, D2, D3, D5 (và I1 vì trùng Goal 2). Augment thuần, gate bằng config, không phá static.
- **Trung bình, để sau**: D4, I2, I4 — giá trị thật nhưng nặng/phụ thuộc bước trước.
- **Không port**: I5 (Thematic Forest), I3-nguyên-bản (Allen vào PPR), I6 (trùng synonymy).
- **Insight cốt lõi**: nền entity-centric + PPR của HippoRAG hấp thụ tốt phần "time-aware retrieval + timeline reasoning" của DyG-RAG; phần "interval-algebra structural" của IA-RAG kén nền PPR — chỉ mượn được ở query-time, nhưng chính nó mở đường cho conflict/supersede (Goal 3/4) của đề tài.

---

## Phụ lục: thông số gốc của hai paper (để đối chiếu khi reproduce)

**DyG-RAG** — DEU = {sentence, normalized timestamp, event_id, source_id}; embedding `z=[BGE-M3 text ; Fourier(time)]`; cạnh khi entity-overlap>0 AND |Δt|≤Δt_thr, weight = Jaccard(entities)·exp(−α|Δt|), giữ top-K neighbor; retrieval: parse t_Q → reweight query (λ) → seed bằng vector+cross-encoder (TinyBERT) → weighted random walk độ dài L → timeline + Time-CoT. Chunk 1200/64, input ≤16384.
Acc vs HippoRAG: TimeQA 58.78 vs 39.99; TempReason 84.75 vs 69.80; ComplexTR 55.62 vs 44.68.

**IA-RAG** — IEU = {sentence, context, [t_start,t_end], fuzzy_flag, source}; trích bằng Qwen2.5-14B; interval từ granularity ngày/tháng/năm/mờ; dedup semantic-neighborhood (τ=0.8); Thematic Forest depth=4 (MEU summarize đệ quy); cạnh khi cos>0.8 + gán 1 trong 13 quan hệ Allen; Sub-graph Time Tightening tinh chỉnh interval mờ bằng LLM; retrieval: parse scope [ws,we] → top-K1 MEU + top-K2 IEU → directional traversal Ψ (before/after) → expand qua Allen → merge. Chunk 1200/64, 4×A100.
Acc vs HippoRAG: TimeQA 61.72 vs 39.99; TempReason 80.21 vs 69.80; ComplexTR 65.95 vs 44.68.
