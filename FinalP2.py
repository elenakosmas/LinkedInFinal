import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


s = pd.read_csv('social_media_usage.csv')

def clean_sm(x):
    return np.where(x == 1, 1, 0)

ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),
    "income": np.where(s["income"].between(1, 9), s["income"], np.nan).astype(float),
    "education": np.where(s["educ2"].between(1, 8), s["educ2"], np.nan).astype(float),
    "parent": np.where(s["par"] == 1, 1, 0),
    "married": np.where(s["marital"] == 1, 1, 0),
    "female": np.where(s["gender"] == 2, 1, 0),
    "age": np.where(s["age"].between(1, 98), s["age"], np.nan).astype(float)
})
ss = ss.dropna()

target_variable = 'sm_li'

y = ss[target_variable]

X = ss.drop(columns=[target_variable])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=987)

logreg_model = LogisticRegression(class_weight='balanced', random_state=987)

logreg_model.fit(X_train, y_train)

y_pred = logreg_model.predict(X_test)

y_prob = logreg_model.predict_proba(X_test)

income_options = {
    "Less than $10k": 1,
    "$10k to under $20k": 2,
    "$20k to under $30k": 3,
    "$30k to under $40k": 4,
    "$40k to under $50k": 5,
    "$50k to under $75k": 6,
    "$75k to under $100k": 7,
    "$100k to under $150k": 8,
    "$150k or more": 9
}

education_options = {
    "Less than high school (Grades 1-8 or no formal schooling)": 1,
    "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)": 2,
    "High school graduate (Grade 12 with diploma or GED certificate)": 3,
    "Some college, no degree (includes some community college)": 4,
    "Two-year associate degree from a college or university": 5,
    "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)": 6,
    "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)": 7,
    "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)": 8
}

parent_options = {
    "Yes": 1,
    "No": 0
}

married_options = {
    "Yes": 1,
    "No": 0
}

gender_options = {
    "Male": 0,
    "Female": 1
}

def predict_linkedin_usage(user_input):
    user_result = logreg_model.predict(user_input)
    user_prob = logreg_model.predict_proba(user_input)[:, 1]  # Probability of class 1
    return user_result, user_prob[0]

import streamlit as st

# Set the page config and styling
st.set_page_config(page_title="LinkedIn Usage Predictor", page_icon=":chart_with_upwards_trend:")

# Change the text and set background color
st.markdown(
    """
    <style>
     .big-text {
            font-size: 26px;
            color: #e75480; /* Dark Pink */
            font-weight: bold;
            text-decoration: underline;
        }
    
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<style>h1{color: #00008B;}</style>', unsafe_allow_html=True)
st.title("LINKEDIN USAGE PREDICTOR")
st.markdown('<div class="big-text">Let\'s check whether you are or not a LinkedIn user!</div>', unsafe_allow_html=True)

# Sidebar with user input options
with st.sidebar:
    st.subheader("Would you mind sharing a few insights first?")

    # Income Level
    st.markdown("<style>div[data-testid='stSelectbox'] > div:first-child { margin-bottom: 0; }</style>", unsafe_allow_html=True)  # Reduce space
    st.markdown("<span style='color: #FF1493; font-weight: bold; margin-bottom: 0.5rem;'>INCOME LEVEL</span>", unsafe_allow_html=True)  # Darker Pink
    income_value = st.selectbox("", list(income_options.keys()), key="income")

    # Education Level
    st.markdown("<style>div[data-testid='stSelectbox'] > div:first-child { margin-bottom: 0; }</style>", unsafe_allow_html=True)  # Reduce space
    st.markdown("<span style='color: #FF1493; font-weight: bold; margin-bottom: 0.5rem;'>EDUCATION LEVEL</span>", unsafe_allow_html=True)  # Dark Pink
    education_value = st.selectbox("", list(education_options.keys()), key="education")

    # Parent
    st.markdown("<style>div[data-testid='stSelectbox'] > div:first-child { margin-bottom: 0; }</style>", unsafe_allow_html=True)  # Reduce space
    st.markdown("<span style='color: #FF1493; font-weight: bold; margin-bottom: 0.5rem;'>ARE YOU A PARENT?</span>", unsafe_allow_html=True)  # Dark Pink
    parent_value = st.selectbox("", list(parent_options.keys()), key="parent")

    # Married
    st.markdown("<style>div[data-testid='stSelectbox'] > div:first-child { margin-bottom: 0; }</style>", unsafe_allow_html=True)  # Reduce space
    st.markdown("<span style='color: #FF1493; font-weight: bold; margin-bottom: 0.5rem;'>MARRIED</span>", unsafe_allow_html=True)  # Dark Pink
    married_value = st.selectbox("", list(married_options.keys()), key="married")

    # Gender
    st.markdown("<style>div[data-testid='stSelectbox'] > div:first-child { margin-bottom: 0; }</style>", unsafe_allow_html=True)  # Reduce space
    st.markdown("<span style='color: #FF1493; font-weight: bold; margin-bottom: 0.5rem;'>GENDER</span>", unsafe_allow_html=True)  # Dark Pink
    gender_value = st.selectbox("", list(gender_options.keys()), key="gender")

    # Age
    st.markdown("<span style='color: #FF1493; font-weight: bold; margin-bottom: 0.5rem;'>WHAT IS YOUR AGE?</span>", unsafe_allow_html=True)  # Dark Pink
    age_value = st.number_input("", min_value=0, max_value=98, value=10, step=1, key="age")

# Create user input DataFrame
user_input = pd.DataFrame({
    "income": [income_options[income_value]],
    "education": [education_options[education_value]],
    "parent": [parent_options[parent_value]],
    "married": [married_options[married_value]],
    "female": [gender_options[gender_value]],
    "age": [age_value]
})

# Predictions and results
user_result, user_prob = predict_linkedin_usage(user_input)

# Display the results with dynamic and colorful messages
result_color = "green" if user_result == 1 else "red"
result_message = f"{'Yes!' if user_result == 1 else 'That is a shame!'} You {'probably' if user_result == 1 else 'probably don’t'} use LinkedIn."
result_emoji = '😊 🎉 ' if user_result == 1 else '😞 👎'
st.markdown(f"## {result_message} {result_emoji}", unsafe_allow_html=True)
st.markdown(f"### Probability of Using LinkedIn: <span style='color:{result_color}'>{user_prob:.2%}</span>", unsafe_allow_html=True)



# Display content based on the predicted LinkedIn usage
if user_result == 1:
    st.markdown("<span style='color: green; font-size: 20px; font-weight: bold;'>Tips for Active LinkedIn Users:</span>", unsafe_allow_html=True)
    st.markdown("- Regularly update your LinkedIn profile.")
    st.markdown("- Connect with professionals in your industry.")
    st.markdown("- Build and nurture your network.")
    st.markdown("- Engage thoughtfully.")
else:
    st.markdown("<span style='color: red; font-size: 20px; font-weight: bold;'>Tips for Enhancing Your LinkedIn Profile:</span>", unsafe_allow_html=True)
    st.markdown("- Complete your LinkedIn profile with a professional photo.")
    st.markdown("- Highlight your skills and experiences.")
    st.markdown("- Write a catchy headline & LinkedIn summary.")

# Display video based on the prediction result
if user_result == 1:
    st.video("https://www.youtube.com/watch?v=OHTRZKg2LS0")
else:
    st.video("https://www.youtube.com/watch?v=Q3bQN4AaM0g")