# Pokédex DL – Hệ thống nhận diện Pokémon (Deep Learning)

## Giới thiệu

Dự án này xây dựng một hệ thống **Nhận diện Pokémon từ ảnh** ứng dụng **Deep Learning (Học sâu)** kết hợp với giao diện người dùng hiện đại. 

Đây là bản nâng cấp toàn diện từ hệ thống Xử lý tín hiệu số (DSP) truyền thống sang việc sử dụng hoàn toàn Mạng Nơ-ron Tích chập (CNN) cho cả quá trình khử nhiễu (Denoise) và phân loại ảnh (Classification). Hệ thống nhận đầu vào là một ảnh và trả về:

* Tên Pokémon dự đoán (Top-3 kết quả)
* Độ tin cậy (Confidence)
* Thông tin chi tiết được lấy theo thời gian thực từ PokéAPI (Loại, Chiều cao, Cân nặng, Ảnh Sprite)

---

## Tính năng nổi bật

1. **Khử nhiễu bằng Trí tuệ nhân tạo:** Thay thế các bộ lọc cổ điển bằng **Convolutional Autoencoder** để làm sạch nhiễu và bảo toàn cấu trúc ảnh.
2. **Nhận diện chính xác cao:** Sử dụng **MobileNetV2** (áp dụng Transfer Learning) giúp mô hình có khả năng trích xuất đặc trưng mạnh mẽ và tốc độ dự đoán cực nhanh.
3. **Giao diện hiện đại (CustomTkinter):** Thiết kế Dark Mode chuyên nghiệp với **Animation Pokéball** tương tác mượt mà.
4. **Hiển thị Pipeline 4 bước (4-Stage Visual):** Người dùng có thể xem toàn bộ quá trình hệ thống xử lý ảnh: *Ảnh gốc -> Ảnh Khử nhiễu (DL) -> Tìm biên (Edges) -> Ảnh cuối cùng*.
5. **Tích hợp PokéAPI:** Tự động gọi API dưới nền (background thread) để hiển thị thông số chi tiết của Pokémon mà không làm đơ giao diện.
6. **Lịch sử tìm kiếm:** Tự động lưu và hiển thị lại các Pokémon đã nhận diện nhờ tích hợp cơ sở dữ liệu SQLite.

---

## Pipeline tổng thể

```
Ảnh đầu vào 
   → Tiền xử lý (Convolutional Autoencoder khử nhiễu + Canny Edge Detection)
   → Mô hình nhận diện (CNN - MobileNetV2)
   → Kết quả (Tên + Độ tin cậy)
   → Ghi log Lịch sử (SQLite)
   → Gọi PokéAPI lấy thông số 
   → Hiển thị lên UI (CustomTkinter)
```

---

## Cấu trúc Project

```
POKEDEXX_DL/
│
├── src/
│   ├── app.py             # Giao diện chính (CustomTkinter)
│   ├── model_dl.py        # Kiến trúc mô hình CNN (MobileNetV2)
│   ├── denoise_dl.py      # Mô hình Convolutional Autoencoder khử nhiễu
│   ├── predict_dl.py      # Hàm suy luận (Inference)
│   ├── train_dl.py        # Script huấn luyện mô hình
│   ├── preprocess.py      # Pipeline xử lý ảnh (OpenCV)
│   ├── dataset.py         # Xử lý DataLoader cho PyTorch
│   ├── api.py             # Gọi dữ liệu từ PokéAPI
│   └── history.py         # Xử lý database SQLite lưu lịch sử
│
├── data/                  # Dataset ảnh (Không bao gồm trên GitHub)
├── Model/                 # Lưu trọng số mô hình đã train (.pth)
├── history.db             # CSDL lưu trữ lịch sử nhận diện
├── README.md              
└── .gitignore
```

---

## Công nghệ sử dụng

* **Deep Learning & Computer Vision:**
  * `PyTorch` / `Torchvision`: Xây dựng và huấn luyện mô hình CNN & Autoencoder.
  * `OpenCV` / `Pillow (PIL)`: Xử lý kích thước, không gian màu và tiền xử lý ảnh.
* **Giao diện (GUI):**
  * `CustomTkinter`: Giao diện đồ họa tối giản, Dark Mode hiện đại.
* **Backend & API:**
  * `Requests`: Lấy thông tin từ PokéAPI.
  * `SQLite3`: Lưu trữ dữ liệu dự đoán cục bộ.
  * `Threading`: Xử lý đa luồng giúp UI mượt mà khi tải mạng.

---

## Lưu ý

* Do giới hạn dung lượng, **Dataset** và **Trọng số mô hình (Weights)** không được đính kèm trực tiếp. Bạn cần tải dataset và chạy script `train_dl.py` để tạo file trọng số.
* Cần đảm bảo có kết nối Internet để ứng dụng có thể lấy được thông tin chi tiết từ PokéAPI.

---

## Tác giả

* **LOCGG1712** – Nâng cấp và phát triển hệ thống Pokedex Deep Learning.

---

## Hướng phát triển tương lai

* Chuyển đổi mô hình sang chuẩn **ONNX** để tối ưu hóa tốc độ chạy suy luận (Inference) trên CPU.
* Mở rộng số lượng class (nhận diện toàn bộ Gen 1 hoặc nhiều hơn).
* Đóng gói dự án thành file thực thi `.exe` độc lập.
