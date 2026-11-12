# Plant Disease Detection API

REST API nhận diện bệnh trên lá cây bằng mô hình YOLO và FastAPI.

## Chạy trên Windows

Mở PowerShell tại thư mục dự án (cần cài Python 3.12), chạy:

```powershell
$envPath = Join-Path $env:TEMP "PlantDiseaseApiVenv"
py -3.12 -m venv $envPath
& "$envPath\Scripts\python.exe" -m pip install -r requirements.txt
& "$envPath\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
```

API chạy tại `http://127.0.0.1:8000`; mở `/docs` để thử API. Dừng bằng `Ctrl+C`.

`/check-leaf` dùng checkpoint YOLOE-26 `yoloe-26l-seg.pt` và prompt
`a leaf from a plant`. Lần đầu cần internet để tải text encoder. Model này chỉ
dùng cho `/check-leaf`; `/predict` vẫn dùng `best.pt`.

## API

- `POST /check-leaf`: kiểm tra ảnh có vùng lá cây hay không. Gửi ảnh dạng
  `multipart/form-data` với trường `file`; có thể truyền `conf` trong khoảng
  `(0, 1]` (mặc định `0.2`). Kết quả trả về `is_leaf` và độ tin cậy của vùng
  lá cao nhất, hoặc `null` nếu không phát hiện lá; `bounding_boxes` chứa các
  bounding box của tất cả vùng lá được phát hiện, rỗng nếu không có.
- `POST /predict`: nhận diện bệnh. API chỉ trả các phát hiện bệnh nếu mô hình
  đồng thời phát hiện vùng lá (`class_id` 0); nếu không, `is_leaf` là `false`,
  trạng thái là `leaf_not_detected`, và danh sách `detections`/`diseases` rỗng.
  Response chỉ gồm thông tin dự đoán; không trả ảnh annotated hoặc bounding box.

Ví dụ kiểm tra ảnh có lá:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/check-leaf?conf=0.2" -F "file=@leaf.jpg"
```

Ví dụ nhận diện bệnh:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict?conf=0.5" -F "file=@leaf.jpg"
```
