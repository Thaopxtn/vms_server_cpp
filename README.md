# Hướng dẫn Xây dựng & Chạy Thử nghiệm VMS C++ Core Server

Tài liệu này hướng dẫn cách thiết lập môi trường, cài đặt các thư viện phụ thuộc, biên dịch và chạy thử nghiệm **VMS C++ Core Server** (phiên bản thương mại hóa Giai đoạn 1).

---

## 1. Các Thư viện Phụ thuộc (Dependencies)

Để biên dịch thành công dự án C++, hệ thống cần được cài đặt sẵn các thư viện sau:

1. **CMake (>= 3.12)** và bộ công cụ compiler hỗ trợ C++17 trở lên (GCC/Clang trên Linux, MSVC/MinGW trên Windows).
2. **OpenCV (>= 4.x):** Cung cấp các tính năng chụp luồng RTSP và tiền xử lý hình ảnh.
3. **ONNX Runtime (C++ API):** Thư viện suy luận AI tốc độ cao.
4. **PostgreSQL Server & pgvector:** Hệ quản trị CSDL quan hệ và tiện ích mở rộng vector.
5. **libpqxx:** Thư viện C++ chính thức kết nối PostgreSQL.
6. **Crow:** Thư viện HTTP Micro-framework header-only cho C++. Cần đặt tệp `crow_all.h` (hoặc thư mục headers) vào trong đường dẫn include.

---

## 2. Các Bước Khởi tạo & Cấu hình

### Bước 2.1: Khởi tạo Cơ sở dữ liệu
Đảm bảo PostgreSQL Server đã chạy và tạo CSDL có tên `smart_monitoring`. Kích hoạt pgvector bằng cách chạy file SQL khởi tạo:

```bash
# Đăng nhập PostgreSQL và chạy file SQL khởi tạo
psql -U postgres -d smart_monitoring -f database/init_postgres.sql
```

### Bước 2.2: Đặt các mô hình AI (.onnx)
Sao chép các mô hình từ ứng dụng Python cũ vào thư mục `models/` của dự án C++:
* `models/yolo11n-face.onnx`
* `models/best_plateSegment.onnx` (cameraAI3)
* `models/det.onnx`, `models/rec.onnx`, `models/ppocrv5_dict.txt` (cameraAI3 OCR)
* `models/face_recognition_sface_2021dec.onnx`

---

## 3. Biên dịch Bằng CMake

### Trên Linux (Ubuntu / Debian):
```bash
# Cài đặt thư viện phụ thuộc trên Ubuntu
sudo apt-get update
sudo apt-get install libopencv-dev libpq-dev libpqxx-dev git cmake g++

# Di chuyển vào thư mục dự án C++
cd vms_server_cpp

# Tạo thư mục build và biên dịch
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Trên Windows:
1. Đảm bảo đã cài đặt Visual Studio (C++ Desktop Development) hoặc MinGW.
2. Tải và cài đặt các thư viện OpenCV, ONNX Runtime, libpqxx tương ứng.
3. Thiết lập biến môi trường chỉ tới thư mục cài đặt (ví dụ: `ONNXRUNTIME_DIR` trỏ tới thư mục chứa thư viện).
4. Sử dụng CMake GUI hoặc VS Code CMake Tools để cấu hình biên dịch sinh file `.exe`.

---

## 4. Chạy Thử nghiệm và Gọi API

Khởi chạy chương trình sau khi biên dịch thành công:
```bash
./vms_server_core
```

Server sẽ lắng nghe các kết nối REST API tại cổng `8080`.

### 4.1: Kiểm tra kết nối Camera
Gửi request thêm camera RTSP vào CSDL và kích hoạt luồng AI:

```bash
curl -X POST http://localhost:8080/api/cameras \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "CAM_TEST_01", "name": "Camera Sanh Truoc", "rtsp_url": "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov"}'
```

### 4.2: Truy vấn danh sách camera
```bash
curl http://localhost:8080/api/cameras
```

### 4.3: Truy xuất danh sách sự kiện AI đã phát hiện
```bash
curl http://localhost:8080/api/events
```
Hoặc lọc theo camera cụ thể:
```bash
curl http://localhost:8080/api/events?camera_id=CAM_TEST_01
```
