# Tiêu chí chấm dự án

| Tiêu chí | Trọng số | Minh chứng để đạt điểm tối đa |
|---|---:|---|
| Kiểm tra dữ liệu và tính toàn vẹn của split | 15 | chia theo source, thống kê identity rate, kiểm tra target noise |
| Câu hỏi nghiên cứu và preregistration | 10 | giả thuyết có hướng và success criterion định lượng |
| Probing có hướng dẫn | 15 | một LM, accuracy và gold candidate score, caveat chính xác |
| Thí nghiệm LoRA có đối chứng | 25 | cùng checkpoint ban đầu, chọn bằng dev CER, cấu hình tái lập được |
| Đánh giá trên locked test | 15 | cùng dòng/cấu hình decoding, CER/WER/chrF theo phương ngữ |
| Downstream recovery | 10 | bảng accuracy cho bốn loại input và phép tính recovery |
| Phân tích lỗi và hạn chế | 10 | ít nhất 10 lỗi được gán nhãn và caveat dựa trên bằng chứng |

Việc dùng kết quả test để điều chỉnh hệ thống làm mất điểm toàn bộ tiêu chí đánh giá
trên locked test.
