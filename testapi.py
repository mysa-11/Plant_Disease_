
import requests
from tkinter import Tk, filedialog

# 1. Chọn ảnh
root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Chọn ảnh lá cây cần nhận diện",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    print("Bạn chưa chọn ảnh!")
    raise SystemExit

# 2. Gửi ảnh lên API
try:
    with open(image_path, "rb") as f:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files={"file": f},
            timeout=120
        )
except requests.exceptions.RequestException as e:
    print("Không kết nối được API:", e)
    raise SystemExit

# 3. Kiểm tra phản hồi
print("HTTP Status:", response.status_code)

try:
    data = response.json()
except ValueError:
    print("API không trả về JSON:")
    print(response.text)
    raise SystemExit

if response.status_code != 200:
    print("API bị lỗi:", response.status_code)
    print("Chi tiết lỗi:", data)
    print("Nội dung phản hồi:", response.text)
    raise SystemExit

if not data.get("success", False):
    print("Nhận diện thất bại:", data)
    raise SystemExit

# 4. Hiển thị kết quả nhận diện
print("\n===== KẾT QUẢ NHẬN DIỆN =====")
print("Tên ảnh:", data.get("filename"))
print("Trạng thái:", data.get("status"))
print("Thông báo:", data.get("message"))
print("Tổng số đối tượng:", data.get("total_detections"))

for item in data.get("detections", []):
    print("\nTên lớp:", item.get("class_name"))
    print("Độ tin cậy:", item.get("confidence"))

print("\nNhận diện thành công!")
