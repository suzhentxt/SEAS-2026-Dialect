# Model Card: Bộ chuẩn hóa phương ngữ SEAS

## Mục đích sử dụng

Mô hình phục vụ giảng dạy học thuật và nghiên cứu thử nghiệm về chuẩn hóa phương ngữ
tiếng Việt sang tiếng Việt chuẩn. Không sử dụng hệ thống để tự động chấm điểm người
học, sửa cách nói của người dùng hoặc đưa ra quyết định có ảnh hưởng lớn.

## Kiến trúc và huấn luyện

Baseline có đối chứng là mBART private (`tarudesu/mbart-large-50`). LoRA adapter của
học viên phải bắt đầu từ cùng checkpoint. Quá trình huấn luyện dùng 192 dòng; việc
chọn checkpoint dùng 48 dòng dev được chia theo source.

## Đánh giá

Metric chính là Character Error Rate, trong đó giá trị thấp hơn tốt hơn. Các metric
bổ sung gồm WER, chrF, exact match, kết quả theo phương ngữ, downstream task recovery
và phân tích lỗi ngữ nghĩa thủ công. NLL recovery chỉ là familiarity diagnostic,
không phải quality metric.

## Hạn chế

Dataset và tập adaptation đều nhỏ; identity pair phân bố không đều giữa các phương
ngữ; reference có thể còn nhiễu; phạm vi chỉ gồm ba nhóm phương ngữ và bốn task nguồn.
