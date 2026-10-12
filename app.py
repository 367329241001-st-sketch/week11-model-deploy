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
    model_files = ["W10-NN-model.pkcls"]

# ตัวเลือกให้ผู้ใช้เลือกโมเดลเองได้
selected_model = st.sidebar.selectbox("ไฟล์โมเดล (*.pkcls)", model_files)

# 2. โหลดไฟล์ .pkcls ด้วย joblib
@st.cache_resource
def load_selected_model(model_name):
    return joblib.load(model_name)

try:
    model = load_selected_model(selected_model)
    st.sidebar.success(f"โหลดโมเดลสำเร็จ: {selected_model}")
except Exception as e:
    st.sidebar.error(f"ไม่สามารถโหลดโมเดล {selected_model} ได้")

# แสดงข้อมูลคลาสที่ทำนายได้บน Sidebar
st.sidebar.write("คลาสที่ทำนายได้: covid, normal, pneumonia")

# ----------------------------------------------------

# 3. แสดงหัวข้อแอปด้วย st.title
st.title("โปรแกรมจำแนกโรค Covid-19 จากภาพ X-ray")

# 4. ส่วนรับไฟล์ภาพ X-ray สำหรับทำนายผล
st.subheader("อัปโหลดภาพ X-ray")
uploaded_file = st.file_uploader("เลือกไฟล์ภาพ (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # แสดงภาพที่ผู้ใช้อัปโหลด
    image = Image.open(uploaded_file)
    st.image(image, caption="ภาพ X-ray ที่อัปโหลด", width=300)

    # 5. ปุ่ม "ทำนายผล" (st.button)
    if st.button("ทำนายผล"):
        try:
            # แปลงไฟล์ภาพให้เป็น Array ตัวเลข ปรับขนาดภาพเป็น 64x64 พิกเซลตามที่ใช้ฝึกใน Colab
            img_resized = image.convert('RGB').resize((64, 64))
            img_array = np.array(img_resized).flatten().reshape(1, -1)

            # ส่งเข้าโมเดลเพื่อทำนายผล
            prediction = model.predict(img_array)[0]

            # 6. แสดงผลการทำนายด้วย st.success
            st.subheader("ผลการทำนาย")
            st.success(f"ผลการจำแนกโรค: **{prediction}**")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดขณะทำนายผล: {e}")
            st.info("💡 ข้อแนะนำ: ตรวจสอบว่าขั้นตอน Preprocessing ภาพ (เช่น ขนาด pixels / RGB) ตรงกับตอนฝึกโมเดลหรือไม่")


    
   
