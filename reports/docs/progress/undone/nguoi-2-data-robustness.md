# 👤 Người 2 — Data & Robustness

- [x] 10–12/11: Dataset audit: class distribution, metadata, duplicate/noisy data.
  → `scripts/eda.py`, kết quả ở `reports/eda/`, tóm tắt ở `data/README.md` (11/11).
- [x] 13–15/11: Chuẩn bị random split + variety/age-aware split.
  → `scripts/prepare_splits.py` → `data/splits/{random,variety,age_old,age_young}/`,
  load bằng `src.data.paddy.build_split_datasets(config)` (11/11).
- [ ] 16–19/11: Chạy robustness tests: variety, age, background, leaf appearance.
- [ ] 20–23/11: Chuẩn bị evaluation data cho proposed method.
- [ ] 24–26/11: Chạy final robustness experiments.
- [ ] 27–28/11: Error analysis + figure dataset/robustness.
- [ ] 29–30/11: Hoàn thiện phần Dataset + Analysis trong report.
