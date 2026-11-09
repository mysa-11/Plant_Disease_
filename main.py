
from pathlib import Path
import json
import base64
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO


# =========================
# 1. ĐƯỜNG DẪN FILE
# =========================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"
CLASS_NAMES_PATH = BASE_DIR / "class_names_vi.json"
DISEASES_PATH = BASE_DIR / "diseases.json"


# =========================
# 2. KIỂM TRA VÀ ĐỌC DỮ LIỆU
# =========================

for path in [MODEL_PATH, CLASS_NAMES_PATH, DISEASES_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = {
        int(k): v for k, v in json.load(f).items()
    }

with open(DISEASES_PATH, "r", encoding="utf-8") as f:
    disease_data = json.load(f)

DISEASES = {
    int(item["ma_nhan"]): item
    for item in disease_data
}


# =========================
# 3. NẠP MÔ HÌNH YOLO
# =========================

model = YOLO(str(MODEL_PATH))

model.model.names = CLASS_NAMES


# =========================
# 4. KHỞI TẠO FASTAPI
# =========================

app = FastAPI(
    title="Plant Disease Detection API",
    description="API nhận diện bệnh trên lá cây bằng YOLO11n",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 5. API TRANG CHỦ
# =========================

@app.get("/")
def home():
    return {
        "message": "Plant Disease Detection API is running",
        "model": "YOLO11n",
        "number_of_classes": 9,
        "supported_diseases": 8
    }


# =========================
# 6. KIỂM TRA API
# =========================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


async def read_uploaded_image(file: UploadFile) -> Image.Image:
    image_bytes = await file.read()

    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="File tải lên không phải ảnh hợp lệ."
        )


def detect_image(image: Image.Image, conf: float):
    results = model.predict(
        source=image,
        imgsz=640,
        conf=conf,
        device="cpu",
        verbose=False
    )
    return results[0]


def has_leaf_detection(result) -> bool:
    return any(int(box.cls[0].item()) == 0 for box in result.boxes)


def validate_confidence(conf: float) -> None:
    if not 0 < conf <= 1:
        raise HTTPException(
            status_code=400,
            detail="Confidence phải nằm trong khoảng (0, 1]."
        )


# =========================
# 7. API KIỂM TRA ẢNH CÓ LÁ
# =========================

@app.post("/check-leaf")
async def check_leaf(
    file: UploadFile = File(...),
    conf: float = 0.5
):
    validate_confidence(conf)
    image = await read_uploaded_image(file)

    try:
        result = detect_image(image, conf)
        leaf_boxes = [
            box for box in result.boxes
            if int(box.cls[0].item()) == 0
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý ảnh: {str(e)}"
        )

    is_leaf = bool(leaf_boxes)
    confidence = (
        round(max(float(box.conf[0].item()) for box in leaf_boxes), 4)
        if is_leaf
        else None
    )

    return {
        "success": True,
        "filename": file.filename,
        "is_leaf": is_leaf,
        "confidence": confidence,
        "status": "leaf_detected" if is_leaf else "leaf_not_detected",
        "message": (
            "Phát hiện vùng lá cây trong ảnh."
            if is_leaf
            else "Không phát hiện được vùng lá cây trong ảnh."
        )
    }


# =========================
# 8. API NHẬN DIỆN BỆNH
# =========================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = 0.5
):
    validate_confidence(conf)
    image = await read_uploaded_image(file)

    try:
        result = detect_image(image, conf)
        leaf_found = has_leaf_detection(result)

        # =========================
        # 9. VẼ BOUNDING BOX
        # =========================

        if leaf_found:
            annotated_image = result.plot()
            annotated_pil = Image.fromarray(
                annotated_image[:, :, ::-1].copy()
            )
        else:
            annotated_pil = image

        image_buffer = BytesIO()

        annotated_pil.save(
            image_buffer,
            format="JPEG",
            quality=90
        )

        annotated_image_base64 = base64.b64encode(
            image_buffer.getvalue()
        ).decode("utf-8")

        # =========================
        # 10. LẤY THÔNG TIN NHẬN DIỆN
        # =========================

        detections = []
        detected_diseases = {}
        boxes_to_process = result.boxes if leaf_found else []

        for box in boxes_to_process:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = CLASS_NAMES.get(
                class_id,
                "Không xác định"
            )

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bounding_box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            }

            # Class 0: vùng lá cây
            if class_id == 0:

                detection["type"] = "leaf"
                detection["disease_info"] = None

            # Class 1-8: các loại bệnh
            elif class_id in DISEASES:

                disease = DISEASES[class_id]

                detection["type"] = "disease"

                detection["disease_info"] = {
                    "ma_nhan": disease["ma_nhan"],
                    "ten_cay": disease["Tên cây trồng"],
                    "ten_benh": disease["Tên bệnh"],
                    "dau_hieu": disease["Dấu hiệu nhận biết"],
                    "nguyen_nhan": disease["Nguyên nhân gây bệnh"],
                    "cach_phong_ngua": disease["Cách phòng ngừa"],
                    "cach_xu_li": disease["Cách xử lí khi cây bị bệnh"]
                }

                detected_diseases[class_id] = detection["disease_info"]

            detections.append(detection)

        # =========================
        # 11. XÁC ĐỊNH TRẠNG THÁI
        # =========================

        disease_found = len(detected_diseases) > 0

        if disease_found:

            status = "disease_detected"
            message = "Đã phát hiện bệnh trên lá cây."

        elif leaf_found:

            status = "no_supported_disease_detected"

            message = (
                "Chưa phát hiện bệnh thuộc 8 loại bệnh mà hệ thống hỗ trợ. "
                "Kết quả này không đồng nghĩa với việc cây hoàn toàn khỏe mạnh."
            )

        else:

            status = "leaf_not_detected"
            message = "Không phát hiện được vùng lá cây trong ảnh."

        # =========================
        # 12. TRẢ KẾT QUẢ API
        # =========================

        return {
            "success": True,
            "filename": file.filename,

            "annotated_image": (
                f"data:image/jpeg;base64,{annotated_image_base64}"
            ),

            "image_size": {
                "width": image.width,
                "height": image.height
            },

            "is_leaf": leaf_found,
            "status": status,
            "message": message,
            "total_detections": len(detections),
            "detections": detections,
            "diseases": list(detected_diseases.values())
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý ảnh: {str(e)}"
        )
