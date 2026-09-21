import streamlit as st
import requests

st.set_page_config(page_title='Bank Churn Predictor', page_icon='🏦')

st.title('Bank Customer Churn Predictor')
st.write('Enter customer details to predict churn probability.')

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=18, max_value=100, value=35)
    tenure = st.number_input('Tenure (years)', min_value=0, max_value=15, value=5)

with col2:
    balance = st.number_input('Balance', min_value=0.0, value=50000.0)
    num_products = st.selectbox('Number of Products', [1, 2, 3, 4])
    has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
    is_active = st.selectbox('Is Active Member', ['Yes', 'No'])
    salary = st.number_input('Estimated Salary', min_value=0.0, value=60000.0)

if st.button('Predict Churn'):
    payload = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_cr_card == 'Yes' else 0,
        'IsActiveMember': 1 if is_active == 'Yes' else 0,
        'EstimatedSalary': salary
    }

    response = requests.post('http://127.0.0.1:8000/predict', json=payload)
    result = response.json()

    probability = result['churn_probability']
    will_churn = result['will_churn']

    st.metric('Churn Probability', f'{probability * 100:.1f}%')

    if will_churn:
        st.error('This customer is likely to churn.')
    else:
        st.success('This customer is likely to stay.')