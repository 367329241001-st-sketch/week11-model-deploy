import streamlit as st
import joblib
import os
from PIL import Image
import numpy as np

# 1. กำหนดแถบด้านข้าง (Sidebar) สำหรับเลือกโมเดล (*.pkcls)
st.sidebar.title("เลือกโมเดล")

# ค้นหาไฟล์ .pkcls ในโฟลเดอร์ปัจจุบัน
model_files = [f for f in os.listdir('.') if f.endswith('.pkcls')]
if not model_files:
    # หากไม่พบไฟล์ ให้ใส่ชื่อไฟล์ตัวอย่างไว้เป็นค่าเริ่มต้น
    model_files = ["W10-NN-model.pkcls"]

# ตัวเลือกให้ผู้ใช้เลือกโมเดลเองได้[cite: 3, 8]
selected_model = st.sidebar.selectbox("ไฟล์โมเดล (*.pkcls)", model_files)

# 2. โหลดไฟล์ .pkcls ด้วย joblib
@st.cache_resource
def load_selected_model(model_name):
    return joblib.load(model_name)

try:
    model = load_selected_model(selected_model)
    st.sidebar.success(f"โหลดโมเดลสำเร็จ: {selected_model}")[cite: 3]
except Exception as e:
    st.sidebar.error(f"ไม่สามารถโหลดโมเดล {selected_model} ได้")

# แสดงข้อมูลคลาสที่ทำนายได้บน Sidebar[cite: 3]
st.sidebar.write("คลาสที่ทำนายได้: covid, normal, pneumonia")[cite: 3]

# ----------------------------------------------------

# 3. แสดงหัวข้อแอปด้วย st.title[cite: 3, 8]
st.title("โปรแกรมจำแนกโรค Covid-19 จากภาพ X-ray")[cite: 3, 8]

# 4. ส่วนรับไฟล์ภาพ X-ray สำหรับทำนายผล[cite: 3]
st.subheader("อัปโหลดภาพ X-ray")[cite: 3]
uploaded_file = st.file_uploader("เลือกไฟล์ภาพ (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])[cite: 3]

if uploaded_file is not None:
    # แสดงภาพที่ผู้ใช้อัปโหลด
    image = Image.open(uploaded_file)
    st.image(image, caption="ภาพ X-ray ที่อัปโหลด", width=300)

    # 5. ปุ่ม "ทำนายผล" (st.button)[cite: 3, 8]
    if st.button("ทำนายผล"):
        try:
            # แปลงไฟล์ภาพให้เป็น Array ตัวเลข เพื่อส่งเข้าโมเดล
            # (หมายเหตุ: ให้ปรับขนาดภาพ (128, 128) ให้ตรงกับขนาดที่เคยใช้ฝึกโมเดล)
            img_resized = image.convert('RGB').resize((128, 128))
            img_array = np.array(img_resized).flatten().reshape(1, -1)

            # ส่งเข้าโมเดลเพื่อทำนายผล[cite: 8]
            prediction = model.predict(img_array)[0]

            # 6. แสดงผลการทำนายด้วย st.success[cite: 3, 8]
            st.subheader("ผลการทำนาย")
            st.success(f"ผลการจำแนกโรค: **{prediction}**")[cite: 3, 8]

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดขณะทำนายผล: {e}")
            st.info("💡 ข้อแนะนำ: ตรวจสอบว่าขั้นตอน Preprocessing ภาพ (เช่น ขนาด pixels / RGB) ตรงกับตอนฝึกโมเดลหรือไม่")


    
   
