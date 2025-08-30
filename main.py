import streamlit as st
from prediction_helper import predict

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Health Insurance Cost Predictor",
    layout="wide",
    page_icon="💡"
)

# -------------------- Header --------------------
st.markdown(
    """
    <div style="background-color:#2C3E50;padding:20px;border-radius:10px;text-align:center;">
        <h1 style="color:white;">🏥 Health Insurance Cost Predictor</h1>
        <p style="color:#ecf0f1;font-size:18px;">
            Estimate your health insurance cost instantly using Machine Learning
        </p>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)

# -------------------- Sidebar --------------------
st.sidebar.title("ℹ️ About this Project")
st.sidebar.info(
    """
    This project predicts health insurance costs based on user details like age, 
    BMI, income, lifestyle, employment, and medical history.  
    It uses a **trained Machine Learning model** that learns patterns from past data 
    and provides an estimated cost.  

    🔗 Connect with me:  
    [LinkedIn](https://www.linkedin.com/in/lavish-here/) | 
    [GitHub](https://github.com/lavishvs)
    """
)

# -------------------- Input Options --------------------
categorical_options = {
    'Gender': ['Male', 'Female'],
    'Marital Status': ['Unmarried', 'Married'],
    'BMI Category': ['Normal', 'Obesity', 'Overweight', 'Underweight'],
    'Smoking Status': ['No Smoking', 'Regular', 'Occasional'],
    'Employment Status': ['Salaried', 'Self-Employed', 'Freelancer'],
    'Region': ['Northwest', 'Southeast', 'Northeast', 'Southwest'],
    'Medical History': [
        'No Disease', 'Diabetes', 'High blood pressure',
        'Diabetes & High blood pressure', 'Thyroid', 'Heart disease',
        'High blood pressure & Heart disease', 'Diabetes & Thyroid',
        'Diabetes & Heart disease'
    ],
    'Insurance Plan': ['Bronze', 'Silver', 'Gold']
}

# -------------------- Form Layout --------------------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age', min_value=18, max_value=100, step=1)
    gender = st.selectbox('Gender', categorical_options['Gender'])
    marital_status = st.selectbox('Marital Status', categorical_options['Marital Status'])
    bmi_category = st.selectbox('BMI Category', categorical_options['BMI Category'])

with col2:
    number_of_dependants = st.number_input('Number of Dependants', min_value=0, max_value=20, step=1)
    income_lakhs = st.number_input('Income in Lakhs', min_value=0, max_value=200, step=1)
    genetical_risk = st.number_input('Genetical Risk (0-5)', min_value=0, max_value=5, step=1)
    employment_status = st.selectbox('Employment Status', categorical_options['Employment Status'])

with col3:
    insurance_plan = st.selectbox('Insurance Plan', categorical_options['Insurance Plan'])
    smoking_status = st.selectbox('Smoking Status', categorical_options['Smoking Status'])
    region = st.selectbox('Region', categorical_options['Region'])
    medical_history = st.selectbox('Medical History', categorical_options['Medical History'])

# -------------------- Input Dictionary --------------------
input_dict = {
    'Age': age,
    'Number of Dependants': number_of_dependants,
    'Income in Lakhs': income_lakhs,
    'Genetical Risk': genetical_risk,
    'Insurance Plan': insurance_plan,
    'Employment Status': employment_status,
    'Gender': gender,
    'Marital Status': marital_status,
    'BMI Category': bmi_category,
    'Smoking Status': smoking_status,
    'Region': region,
    'Medical History': medical_history
}

# -------------------- Prediction Button --------------------
if st.button('🔮 Predict My Insurance Cost', use_container_width=True):
    prediction = predict(input_dict)
    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg,#6dd5ed,#2193b0);
                    padding:20px;border-radius:12px;text-align:center;">
            <h2 style="color:white;">💰 Estimated Insurance Cost</h2>
            <h1 style="color:#FFD700;">₹ {prediction:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------- Footer --------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; padding:15px;">
        <p>
            📘 <b>How it Works:</b><br>
            1️⃣ You provide your details (age, income, lifestyle, health history). <br>
            2️⃣ These inputs are transformed into numerical features. <br>
            3️⃣ A pre-trained Machine Learning model processes the features. <br>
            4️⃣ It predicts an estimated insurance cost based on past data patterns.  
        </p>
        <p>
            🔗 <b>Connect with me:</b>  
            <a href="https://www.linkedin.com/in/lavish-here/" target="_blank">LinkedIn</a> | 
            <a href="https://github.com/lavishvs" target="_blank">GitHub</a>
        </p>
        <p>
            🚀 Built with ❤️ using <b>Python</b> & <b>Streamlit</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
