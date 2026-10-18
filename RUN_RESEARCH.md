# Chạy tiếp nghiên cứu Phase 4A–6

Mở Terminal và chạy:

```bash
cd "/Users/macrisen/Desktop/credit card fraud detect"
.venv_defender/bin/python scripts/run_remaining_phases.py --phase all --resume
```

Không cần `source activate`: lệnh dùng trực tiếp Python của môi trường dự án.
Launcher tự cấu hình thư viện OpenMP trên macOS.
Chỉ chạy một tiến trình tại một thời điểm để tránh ghi cùng checkpoint.

Để lưu log và xem tiến độ trong một Terminal khác:

```bash
.venv_defender/bin/python scripts/run_remaining_phases.py --phase all --resume > reports/execution_control/manual_resume.log 2>&1
```

```bash
tail -f reports/execution_control/manual_resume.log
```

`Ctrl+C` trong Terminal chạy `tail` chỉ đóng phần xem log.
Máy cần thức trong lúc chạy. Có thể thêm `caffeinate -i` trước lệnh Python trên macOS.

Chạy riêng từng phase:

```bash
.venv_defender/bin/python scripts/run_remaining_phases.py --phase 4a --resume
.venv_defender/bin/python scripts/run_remaining_phases.py --phase 4b --resume
.venv_defender/bin/python scripts/run_remaining_phases.py --phase 4c --resume
.venv_defender/bin/python scripts/run_remaining_phases.py --phase 5 --resume
.venv_defender/bin/python scripts/run_remaining_phases.py --phase 6 --resume
```

Phase sau yêu cầu phase trước có trạng thái `PASS`. `PASS` nghĩa là thực thi và
kiểm tra hoàn tất, không có nghĩa mô hình RL tốt hơn baseline.
Không đổi config hoặc seed khi tiếp tục cùng bộ thí nghiệm.

- Phase 4A/4B: khôi phục checkpoint huấn luyện đã lưu; bước chưa checkpoint có thể chạy lại.
- Phase 4C: nếu bị ngắt trước khi hoàn tất, ma trận đánh giá chạy lại.
- Phase 5: tái sử dụng các kịch bản đã hoàn tất; kịch bản dở dang được replay có kiểm tra provenance.
- Phase 6: tái sử dụng các lượt final evaluation đã lưu.

Kết quả: `reports/phase4a`, `reports/phase4b`, `reports/phase4c`, `reports/phase5`,
`reports/final_research`. Trạng thái mỗi phase nằm trong `run_manifest.json`.
Nếu có `PARTIAL`, đọc `failure.log`; không sửa trạng thái thành `PASS` bằng tay.
Model RL nằm trong `models/rl`; Defender cập nhật nằm trong `models/continual_defender`.
Các bảng PostgreSQL và Defender baseline gốc không bị ghi đè bởi launcher này.
