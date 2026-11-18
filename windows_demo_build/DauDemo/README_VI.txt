DẤU DEMO — WINDOWS
=================

Yêu cầu: Windows 10/11 x64 (Intel/AMD), RAM khuyến nghị 8 GB, vài GB đĩa trống.
Không cần GPU. Cần Python 3.10–3.12 x64 có sẵn (Python 3.12.10 phù hợp).
Windows ARM chưa được kiểm chứng.

1. Giải nén toàn bộ ZIP (không chạy trực tiếp bên trong ZIP).
2. Trong thư mục DauDemo, nhấp đúp run.cmd; hoặc mở terminal tại đó và chạy:

   .\run.cmd

Launcher ưu tiên Python có sẵn qua py -3.12, py -3.11, py -3.10 hoặc python.
Không tải hoặc cài lại Python. Tự tạo .venv trong thư mục demo, dùng Python
đã có để cài NumPy và PyTorch CPU. Nếu .venv tương thích đã có, sẽ dùng lại.
Lần đầu cần Internet để cài thư viện còn thiếu. Không sửa PATH hoặc thư viện
của project khác. Nếu không tìm thấy Python phù hợp, launcher báo lỗi rõ ràng.
Nếu Windows yêu cầu xác nhận/chặn ứng dụng, đọc thông báo trước khi cho phép.
Thư viện có thể tải vài trăm MB. Không đóng cửa sổ khi đang cài.

Sau khi nạp model, trình duyệt tự mở địa chỉ 127.0.0.1 với cổng còn trống.
Nếu trình duyệt không tự mở, copy địa chỉ Ready trong cửa sổ terminal.
Không mở index.html trực tiếp: cần server để gọi model.

Gõ không dấu, nhấn Space để cập nhật; chữ sửa sáng trong 1 giây.
Nút Thêm dấu xử lý thủ công. Tối đa 256 token chữ/số mỗi lượt.
Giữ cửa sổ terminal mở; Ctrl+C để tắt. Lần sau vẫn chạy run.cmd.
Sau khi cài đủ, inference không cần Internet và không gửi văn bản ra ngoài.
Server chỉ lắng nghe 127.0.0.1, không công khai ra mạng LAN.

Model: punctuation-aware CRF schema2; best checkpoint sau 2 epoch đầy đủ và
một phần epoch 3, chưa phải model đã train đủ 4 epoch. Có thể chọn sai dấu
hoặc thay từ đã có dấu: luôn đọc lại kết quả. Dấu câu không được tự tạo.

Đã kiểm thử inference và HTTP bằng gói portable trên Linux. Launcher Windows
đã được chuẩn bị nhưng chưa chạy thực tế trên Windows trong môi trường này.
Nếu lỗi, gửi nội dung lỗi terminal. Nếu báo thiếu DLL PyTorch, có thể cần
Microsoft Visual C++ Redistributable x64 chính thức từ Microsoft.

Nguồn dữ liệu huấn luyện: Wikimedia Wikipedia 20231101.vi,
https://huggingface.co/datasets/wikimedia/wikipedia
Nội dung nguồn theo CC-BY-SA 3.0/GFDL; gói không chứa văn bản corpus.
Không phân phối gói như một sản phẩm có bảo đảm; phục vụ demo dự án.
