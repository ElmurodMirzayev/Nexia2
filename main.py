# opencv-python==4.13.0.92
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from PIL import Image
import tempfile
import cv2

st.title("Oddiy YOLO Web App")

page = st.sidebar.radio("Sahifani tanlang", ["Detect Model", "Segmentation Model"])

# Modellarni yuklash
model_detect = YOLO("buzilgan_detalniy_detect/train1/weights/best.pt")
model_segment = YOLO("small_model_seg_180ta_datalik_Nexia2damage/train1/weights/best.pt")

uploaded_file = st.file_uploader("Rasm yuklang", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original rasm")

    if st.button("Predict"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        if page == "Detect Model":
            # Threshold 70%
            results = model_detect.predict(source=temp_path, conf=0.7)
            result_image = results[0].plot()
            st.image(result_image, caption="Natija")

        else:
            results = model_segment.predict(source=temp_path)

            # Faqat maska (boxsiz)
            result_image = results[0].plot(boxes=False)

            names = results[0].names
            boxes = results[0].boxes

            if boxes is not None:

                h, w = result_image.shape[:2]

                base_w = 800
                base_h = 600

                scale_w = w / base_w
                scale_h = h / base_h
                scale = min(scale_w, scale_h)

                scale = max(0.5, min(scale, 1.5))

                font_scale = 0.8 * scale
                thickness = max(2, int(3 * scale))
                y_offset = int(10 * scale)

                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = names[cls_id]

                    confidence = float(box.conf[0])
                    label = f"{class_name} {confidence*100:.1f}%"

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    color = colors(cls_id, True)

                    # 🔥 Matn o'lchamini olish
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        thickness
                    )

                    # 🔥 Qora fon chizish
                    cv2.rectangle(
                        result_image,
                        (x1, y1 - text_h - baseline),
                        (x1 + text_w, y1),
                        (0, 0, 0),   # qora rang
                        -1           # to'ldirilgan
                    )

                    # 🔥 Matnni yozish
                    cv2.putText(
                        result_image,
                        label,
                        (x1, y1 - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        thickness
                    )

            st.image(result_image, caption="Natija")