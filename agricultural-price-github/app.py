import streamlit as st
import pandas as pd
import joblib
from datetime import date

st.set_page_config(
    page_title="Agricultural Modal Price Prediction",
    page_icon="🌾",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Price_Agriculture_commodities_Week.csv")

model = load_model()
df = load_data()

st.title("🌾 Agricultural Commodity Modal Price Prediction")
st.write(
    "An intelligent web application for predicting the modal price "
    "of an agricultural commodity using the final Linear Regression model."
)

st.divider()

st.subheader("1. Commodity Information")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State", sorted(df["State"].dropna().astype(str).unique()))
    district = st.selectbox("District", sorted(df["District"].dropna().astype(str).unique()))
    market = st.selectbox("Market", sorted(df["Market"].dropna().astype(str).unique()))
    commodity = st.selectbox("Commodity", sorted(df["Commodity"].dropna().astype(str).unique()))
    variety = st.selectbox("Variety", sorted(df["Variety"].dropna().astype(str).unique()))

with col2:
    grade = st.selectbox("Grade", sorted(df["Grade"].dropna().astype(str).unique()))
    arrival_date = st.date_input("Arrival Date", value=date(2023, 7, 27))
    min_price = st.number_input("Minimum Price", min_value=0.0, value=2200.0, step=100.0)
    max_price = st.number_input("Maximum Price", min_value=0.0, value=3000.0, step=100.0)

st.divider()

st.subheader("2. Prediction")

if st.button("🔮 Predict Modal Price", type="primary", use_container_width=True):
    if min_price > max_price:
        st.error("Minimum Price cannot be greater than Maximum Price.")
    else:
        dt = pd.Timestamp(arrival_date)

        sample = pd.DataFrame([{
            "State": state,
            "District": district,
            "Market": market,
            "Commodity": commodity,
            "Variety": variety,
            "Grade": grade,
            "Year": dt.year,
            "Month": dt.month,
            "Day": dt.day,
            "DayOfWeek": dt.dayofweek,
            "WeekOfYear": int(dt.isocalendar().week),
            "Min Price": min_price,
            "Max Price": max_price,
        }])

        prediction = float(model.predict(sample)[0])

        st.success("Prediction completed successfully!")
        st.metric("Predicted Modal Price", f"{prediction:,.2f}")

        with st.expander("Show model input representation"):
            st.dataframe(sample, use_container_width=True)

st.divider()
st.caption("Model: Linear Regression | Preprocessing: One-Hot Encoding + StandardScaler")
