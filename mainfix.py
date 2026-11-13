
from pathlib import Path
import json
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO, YOLOE


# =========================
# 1. ĐƯỜNG DẪN FILE
# =========================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"
LEAF_MODEL_PATH = BASE_DIR / "yoloe-26l-seg.pt"
LEAF_PROMPTS = ["a leaf from a plant"]
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

leaf_model = None


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
        "number_of_classes": 11,
        "supported_diseases": 8,
        "healthy_classes": 3
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


def classify_image(image):
    results = model.predict(
        source=image,
        imgsz=224,
        device="cpu",
        verbose=False
    )
    return results[0]


def detect_leaf_image(image: Image.Image, conf: float):
    global leaf_model

    if leaf_model is None:
        if not LEAF_MODEL_PATH.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"Không tìm thấy model nhận diện lá: {LEAF_MODEL_PATH}"
            )

        prompted_model = YOLOE(str(LEAF_MODEL_PATH))
        prompted_model.set_classes(LEAF_PROMPTS)
        leaf_model = prompted_model

    results = leaf_model.predict(
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
    conf: float = 0.2
):
    validate_confidence(conf)
    image = await read_uploaded_image(file)

    try:
        result = detect_leaf_image(image, conf)
        leaf_boxes = [
            box for box in result.boxes
            if int(box.cls[0].item()) == 0
        ]
        bounding_boxes = [
            {
                "x1": round(float(box.xyxy[0][0].item()), 2),
                "y1": round(float(box.xyxy[0][1].item()), 2),
                "x2": round(float(box.xyxy[0][2].item()), 2),
                "y2": round(float(box.xyxy[0][3].item()), 2)
            }
            for box in leaf_boxes
        ]
    except HTTPException:
        raise
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
        "bounding_boxes": bounding_boxes,
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
        # =========================
        # 9. LẤY KẾT QUẢ CLASSIFICATION
        # =========================

        result = classify_image(image)

        if result.probs is None:
            raise HTTPException(
                status_code=500,
                detail="Model hiện tại không trả về kết quả Classification. "
                       "Hãy kiểm tra lại file best.pt."
            )

        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf.item())

        class_name = CLASS_NAMES.get(
            class_id,
            "Không xác định"
        )

        print("\n========== KẾT QUẢ CLASSIFICATION ==========")
        print("Class ID:", class_id)
        print("Class name:", class_name)
        print("Confidence:", confidence)

        # =========================
        # 10. LẤY THÔNG TIN LỚP
        # =========================

        disease = DISEASES.get(class_id)

        if disease is None:
            raise HTTPException(
                status_code=500,
                detail=f"Không tìm thấy thông tin cho class_id = {class_id} "
                       "trong diseases.json."
            )

        print("Disease data:", disease)
        print("============================================")

        # =========================
        # 11. XÁC ĐỊNH LỚP KHỎE MẠNH
        # =========================

        folder_name = str(
            disease.get("folder_goc", "")
        ).lower()

        class_folder = str(
            disease.get("ten_lop", "")
        ).lower()

        disease_name = str(
            disease.get("Tên bệnh", "")
        ).lower()

        is_healthy = (
            "healthy" in folder_name
            or "healthy" in class_folder
            or "khoe manh" in folder_name
            or "khoe manh" in class_folder
            or "khỏe mạnh" in disease_name
            or "khoe manh" in disease_name
        )

        # =========================
        # 12. LẤY THÔNG TIN BỆNH
        # =========================

        disease_info = {
            "ma_nhan": disease["ma_nhan"],
            "ten_cay": disease["Tên cây trồng"],
            "ten_benh": disease["Tên bệnh"],
            "dau_hieu": disease["Dấu hiệu nhận biết"],
            "nguyen_nhan": disease["Nguyên nhân gây bệnh"],
            "cach_phong_ngua": disease["Cách phòng ngừa"],
            "cach_xu_li": disease["Cách xử lí khi cây bị bệnh"]
        }

        # =========================
        # 13. XÁC ĐỊNH TRẠNG THÁI
        # =========================

        if confidence < conf:

            status = "low_confidence"

            message = (
                "Độ tin cậy của kết quả còn thấp. "
                "Vui lòng chụp lại ảnh lá rõ nét và đủ sáng."
            )

            returned_disease_info = None

        elif is_healthy:

            status = "healthy"

            message = "Lá cây được phân loại vào nhóm khỏe mạnh."

            returned_disease_info = disease_info

        else:

            status = "disease_detected"

            message = "Đã nhận diện được loại bệnh trên lá cây."

            returned_disease_info = disease_info

        # =========================
        # 14. TRẢ KẾT QUẢ API
        # =========================

        print("Status:", status)
        print("============================================\n")

        return {
            "success": True,
            "filename": file.filename,
            "status": status,
            "message": message,
            "prediction": {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4)
            },
            "disease_info": returned_disease_info
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý ảnh: {str(e)}"
        )
