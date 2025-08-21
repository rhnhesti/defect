import streamlit as st
import logging
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import supervision as sv
import json
from ultralytics import YOLO

# Set the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.WARNING)
st.set_page_config(page_title="AI Object Detection", page_icon="🤖")

# Define the zone polygon
zone_polygon_m = np.array([[160, 100],
                           [160, 380],
                           [481, 380],
                           [481, 100]], dtype=np.int32)

# Initialize the YOLOv8 model


@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)


# Load the YOLO model (this will be cached)
model = load_yolo_model("bestslag.pt")  # Ganti "best.pt" dengan model Anda

# Initialize the tracker, annotators, and zone
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


def save_detections_to_json(detections, file_name):
    detections_data = [{
        "box": box.tolist(),
        "confidence": float(conf),
        "class_id": int(class_id)
    } for box, conf, class_id in zip(detections.xyxy, detections.confidence, detections.class_id)]

    json_str = json.dumps(detections_data)
    st.download_button(label="Download JSON", data=json_str,
                       file_name=file_name, mime="application/json")


def main():
    st.title("🤖 YOLOvWeld")
    st.subheader("YOLOv8 & Streamlit Web Integration")
    st.sidebar.title("Select an option ⤵️")
    choice = st.sidebar.radio(
        "", ("Capture Image And Predict", ":rainbow[Multiple Images Upload -]🖼️🖼️🖼️"), index=1)
    conf = st.slider("Score threshold", 0.0, 1.0, 0.3, 0.05)

    if choice == "Capture Image And Predict":
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(
                bytes_data, np.uint8), cv2.IMREAD_COLOR)

            results = model.predict(cv2_img)
            results1 = results[0] if isinstance(results, list) else results

            detections = sv.Detections.from_ultralytics(results1)
            detections = detections[detections.confidence > conf]
            labels = [
                f"#{index + 1}: {results1.names[class_id]}"
                for index, class_id in enumerate(detections.class_id)
            ]

            labels1 = [
                f"#{index + 1}: {results1.names[class_id]} (Accuracy: {detections.confidence[index]:.2f})"
                for index, class_id in enumerate(detections.class_id)]

            annotated_frame1 = box_annotator.annotate(
                cv2_img, detections=detections)
            annotated_frame1 = label_annotator.annotate(
                annotated_frame1, detections=detections, labels=labels)

            st.image(annotated_frame1, channels="BGR")
            st.write(':orange[ Info : ⤵️ ]')
            st.json(labels1)

            # Convert annotated image to bytes for download
            img = Image.fromarray(cv2.cvtColor(
                annotated_frame1, cv2.COLOR_BGR2RGB))
            buf = BytesIO()
            img.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            # Button to download image
            st.download_button(label="Download Image with Detections",
                               data=byte_im, file_name="detected_image.jpg", mime="image/jpeg")

            # Save detections to JSON
            save_detections_to_json(detections, "detections.json")

    elif choice == ":rainbow[Multiple Images Upload -]🖼️🖼️🖼️":
        uploaded_files = st.file_uploader(
            "Choose images", type=['png', 'jpg', 'webp', 'bmp'], accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            cv2_img = cv2.imdecode(np.frombuffer(
                bytes_data, np.uint8), cv2.IMREAD_COLOR)

            results = model.predict(cv2_img)
            results1 = results[0] if isinstance(results, list) else results

            detections = sv.Detections.from_ultralytics(results1)
            detections = detections[detections.confidence > conf]
            labels = [
                f"#{index + 1}: {results1.names[class_id]}"
                for index, class_id in enumerate(detections.class_id)
            ]

            labels1 = [
                f"#{index + 1}: {results1.names[class_id]} (Accuracy: {detections.confidence[index]:.2f})"
                for index, class_id in enumerate(detections.class_id)
            ]

            annotated_frame1 = box_annotator.annotate(
                cv2_img, detections=detections)
            annotated_frame1 = label_annotator.annotate(
                annotated_frame1, detections=detections, labels=labels)

            st.image(annotated_frame1, channels="BGR")
            st.write(':orange[ Info : ⤵️ ]')
            st.json(labels1)

            # Convert annotated image to bytes for download
            img = Image.fromarray(cv2.cvtColor(
                annotated_frame1, cv2.COLOR_BGR2RGB))
            buf = BytesIO()
            img.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            # Button to download image
            st.download_button(label=f"Download {uploaded_file.name} with Detections",
                               data=byte_im, file_name=f"{uploaded_file.name}_detected.jpg", mime="image/jpeg")

            # Save detections to JSON
            save_detections_to_json(
                detections, f"{uploaded_file.name}_detections.json")

    st.write(':orange[ Classes : ⤵️ ]')
    cls_name = model.names
    cls_lst = list(cls_name.values())
    st.write(f':orange[{cls_lst}]')


if __name__ == '__main__':
    main()

# streamlit run app.py --server.enableXsrfProtection false
