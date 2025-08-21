# app.py
import supervision as sv
from ultralytics import YOLO
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import logging
import json
import os
# cegah error duplikasi lib (intel MKL/OpenMP)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ------------------------------- #
# Konfigurasi dasar
# ------------------------------- #
logging.basicConfig(level=logging.WARNING)
st.set_page_config(page_title="YOLOvWeld", page_icon="🤖", layout="centered")

# ------------------------------- #
# Utilitas
# ------------------------------- #


@st.cache_resource
def load_yolo_model(model_path: str):
    """Load YOLO model sekali (cache)"""
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()
    return model


def bytes_to_cv2_image(bytes_data: bytes):
    """Konversi bytes -> ndarray BGR (cv2)"""
    arr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gambar tidak valid / gagal didekode.")
    return img  # BGR


def annotate_image(image_bgr: np.ndarray, detections: sv.Detections, labels: list):
    """Gambar bbox + label ke citra"""
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated = box_annotator.annotate(image_bgr.copy(), detections=detections)
    annotated = label_annotator.annotate(
        annotated, detections=detections, labels=labels)
    # Tambah jumlah objek di kiri-atas
    count_text = f"Objects in Frame: {len(detections)}"
    cv2.putText(annotated, count_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    return annotated


def detections_to_json(detections: sv.Detections, class_names: dict):
    """Konversi ke list dict (JSON-serializable)"""
    out = []
    xyxy = detections.xyxy  # (N, 4)
    conf = detections.confidence  # (N,)
    cls_ids = detections.class_id  # (N,)
    for i in range(len(detections)):
        box = xyxy[i].tolist()
        c = float(conf[i]) if conf is not None else None
        cid = int(cls_ids[i]) if cls_ids is not None else None
        out.append({
            "box": box,                         # [x1, y1, x2, y2]
            "confidence": c,
            "class_id": cid,
            "class_name": class_names.get(cid, str(cid))
        })
    return out


def cv2_to_download_bytes(image_bgr: np.ndarray, fmt: str = "JPEG"):
    """Konversi BGR -> bytes (untuk download)"""
    img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ------------------------------- #
# Load model (ganti nama file sesuai punyamu)
# ------------------------------- #
MODEL_PATH = "bestslag.pt"  # <--- ganti jika perlu
model = load_yolo_model(MODEL_PATH)

# ------------------------------- #
# UI
# ------------------------------- #
st.title("🤖 YOLOvWeld Detection")
st.subheader("YOLOv8 + Streamlit")
st.sidebar.title("Pilih Mode")
mode = st.sidebar.radio(
    "", ("Capture Image & Predict", "Multiple Images Upload"))
conf_thres = st.slider("Score threshold", 0.0, 1.0, 0.3, 0.05)

# ------------------------------- #
# MODE 1: Kamera
# ------------------------------- #
if mode == "Capture Image & Predict":
    img_file_buffer = st.camera_input("Ambil gambar dari kamera")
    if img_file_buffer is not None:
        try:
            bytes_data = img_file_buffer.getvalue()
            img_bgr = bytes_to_cv2_image(bytes_data)

            # Inference
            results = model.predict(img_bgr, conf=conf_thres, verbose=False)
            res = results[0] if isinstance(results, list) else results

            # Supervision Detections
            detections = sv.Detections.from_ultralytics(res)
            # filter ulang agar pasti sesuai slider
            if detections.confidence is not None:
                detections = detections[detections.confidence > conf_thres]

            # Labels: "#i: class (conf)"
            labels = []
            names = res.names  # dict id->name
            for i, cid in enumerate(detections.class_id or []):
                cname = names.get(int(cid), str(cid))
                cval = float(
                    detections.confidence[i]) if detections.confidence is not None else 0.0
                labels.append(f"#{i+1}: {cname} (Accuracy: {cval:.2f})")

            # Annotate & Tampilkan
            annotated = annotate_image(img_bgr, detections, labels)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Hasil Deteksi", use_container_width=True)

            # Info JSON
            st.write(':orange[Info Deteksi:]')
            st.json(labels if labels else ["Tidak ada objek terdeteksi"])
            json_data = json.dumps(detections_to_json(
                detections, names), ensure_ascii=False, indent=2)
            st.download_button("Download JSON Deteksi", json_data,
                               file_name="detections.json", mime="application/json")

            # Download gambar anotasi
            img_bytes = cv2_to_download_bytes(annotated, fmt="JPEG")
            st.download_button("Download Gambar dengan Deteksi", img_bytes,
                               file_name="detected_image.jpg", mime="image/jpeg")
        except Exception as e:
            st.error(f"Terjadi error saat memproses gambar: {e}")

# ------------------------------- #
# MODE 2: Multi Upload
# ------------------------------- #
elif mode == "Multiple Images Upload":
    uploaded_files = st.file_uploader("Pilih gambar", type=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
                                      accept_multiple_files=True)
    if uploaded_files:
        for uf in uploaded_files:
            st.markdown("---")
            st.write(f"**File:** {uf.name}")
            try:
                bytes_data = uf.read()  # cukup sekali read
                img_bgr = bytes_to_cv2_image(bytes_data)

                # Inference
                results = model.predict(
                    img_bgr, conf=conf_thres, verbose=False)
                res = results[0] if isinstance(results, list) else results

                # Supervision Detections
                detections = sv.Detections.from_ultralytics(res)
                if detections.confidence is not None:
                    detections = detections[detections.confidence > conf_thres]

                # Labels
                labels = []
                names = res.names
                for i, cid in enumerate(detections.class_id or []):
                    cname = names.get(int(cid), str(cid))
                    cval = float(
                        detections.confidence[i]) if detections.confidence is not None else 0.0
                    labels.append(f"#{i+1}: {cname} (Accuracy: {cval:.2f})")

                # Annotate & tampilkan
                annotated = annotate_image(img_bgr, detections, labels)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Hasil: {uf.name}",
                         use_container_width=True)

                # Info deteksi + unduhan
                st.write(':orange[Info Deteksi:]')
                st.json(labels if labels else ["Tidak ada objek terdeteksi"])

                json_data = json.dumps(detections_to_json(
                    detections, names), ensure_ascii=False, indent=2)
                st.download_button(f"Download JSON ({uf.name})", json_data,
                                   file_name=f"{uf.name}_detections.json", mime="application/json")

                img_bytes = cv2_to_download_bytes(annotated, fmt="JPEG")
                st.download_button(f"Download Gambar ({uf.name})", img_bytes,
                                   file_name=f"{uf.name}_detected.jpg", mime="image/jpeg")
            except Exception as e:
                st.error(f"Gagal memproses {uf.name}: {e}")
    else:
        st.info("Unggah satu atau beberapa gambar untuk mulai deteksi.")


# streamlit run app.py --server.enableXsrfProtection false
