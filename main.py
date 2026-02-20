import streamlit as st
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from PIL import Image
import tempfile
import cv2

st.title("Oddiy YOLO Web App")

page = st.sidebar.radio("Sahifani tanlang", ["Detect Model", "Segmentation Model"])

model_detect = YOLO("buzilgan_detalniy_detect/train1/weights/best.pt")
model_segment = YOLO("oxirgisi_shu/weights/best.pt")

uploaded_file = st.file_uploader("Rasm yuklang", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original rasm")

    if st.button("Predict"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        if page == "Detect Model":
            results = model_detect.predict(source=temp_path)
            result_image = results[0].plot()
            st.image(result_image, caption="Natija")

        else:
            results = model_segment.predict(source=temp_path)

            # Faqat maska
            result_image = results[0].plot(boxes=False)

            names = results[0].names
            boxes = results[0].boxes

            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = names[cls_id]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # YOLO ishlatgan rangni olamiz
                    color = colors(cls_id, True)  # BGR format

                    cv2.putText(
                        result_image,
                        class_name,
                        (x1, y1 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        color,
                        4
                    )

            st.image(result_image, caption="Natija")