import streamlit as st
import pandas as pd
import joblib

# 1. โหลดไฟล์ model.pkl ด้วย joblib ตอนเริ่มแอป
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

# เรียกใช้งานการโหลดโมเดล
try:
    model = load_model()
except Exception as e:
    st.error(f"ไม่สามารถโหลดไฟล์ model.pkl ได้ กรุณาตรวจสอบว่าวางไฟล์ไว้ในโฟลเดอร์เดียวกันหรือไม่: {e}")
    st.stop()

# 2. แสดงหัวข้อแอปด้วย st.title
st.title("ระบบวิเคราะห์และทำนายการยกเลิกบริการลูกค้า (Customer Churn Prediction)")
st.write("กรุณากรอกข้อมูลลูกค้าเพื่อทำนายแนวโน้มการยกเลิกบริการ")

# 3. สร้างช่องให้ผู้ใช้กรอกค่าตัวแปรต้น (features)
st.subheader("กรอกข้อมูลตัวแปรต้น (Features)")

# ตัวอย่างช่องกรอกข้อมูลตามชนิดของคอลัมน์
gender = st.selectbox("เพศ (gender)", ["Male", "Female"])
senior_citizen = st.selectbox("เป็นผู้สูงอายุหรือไม่ (SeniorCitizen)", [0, 1])
partner = st.selectbox("มีคู่สมรสหรือไม่ (Partner)", ["Yes", "No"])
tenure = st.number_input("ระยะเวลาที่เป็นลูกค้า/เดือน (tenure)", min_value=0, max_value=120, value=12)
contract = st.selectbox("ประเภทสัญญา (Contract)", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("ใบแจ้งหนี้แบบไม่ใช้กระดาษ (PaperlessBilling)", ["Yes", "No"])
monthly_charges = st.number_input("ค่าบริการรายเดือน (MonthlyCharges)", min_value=0.0, value=70.0, step=0.5)
total_charges = st.number_input("ค่าบริการรวมทั้งหมด (TotalCharges)", min_value=0.0, value=840.0, step=1.0)

# 4. มีปุ่ม "ทำนายผล" (st.button) เมื่อกดแล้วให้นำค่าไปจัดรูปแบบแล้วส่งเข้าโมเดล
if st.button("ทำนายผล"):
    # รวมข้อมูลที่กรอกเข้ามาเป็น DataFrame 1 แถว
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'tenure': tenure,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    # ทำ One-Hot Encoding ข้อมูลข้อความให้เหมือนตอนฝึกโมเดล
    input_encoded = pd.get_dummies(input_data)

    # จัดระเบียบคอลัมน์ให้ตรงกับที่โมเดลเคยใช้ตอนฝึก (เติมคอลัมน์ที่ขาดด้วยค่า 0)
    if hasattr(model, 'feature_names_in_'):
        expected_cols = model.feature_names_in_
        input_encoded = input_encoded.reindex(columns=expected_cols, fill_value=0)

    # ส่งข้อมูลเข้าโมเดลเพื่อทำนายผล[cite: 7]
    prediction = model.predict(input_encoded)[0]

    # 5. แสดงผลการทำนายด้วย st.success หรือ st.warning[cite: 7]
    st.subheader("ผลการทำนาย")
    if prediction == 'Yes' or prediction == 1:
        st.warning("⚠️ ลูกค้ามีแนวโน้มที่จะ **ยกเลิกบริการ (Churn)**")
    else:
        st.success("✅ ลูกค้ามีแนวโน้มที่จะ **ใช้งานต่อ (No Churn)**")
