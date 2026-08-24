# Agricultural Commodity Modal Price Prediction

A Streamlit web application that deploys the final Linear Regression model from Assignment 01.

## Pipeline

User Input → Feature Representation → Preprocessing → Linear Regression → Prediction → Web Output

## Files

- `app.py` — Streamlit web application
- `final_model.pkl` — trained model pipeline, including One-Hot Encoding, StandardScaler, and Linear Regression
- `Price_Agriculture_commodities_Week.csv` — dataset used to populate categorical input options
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Push these files to GitHub and deploy `app.py` using Streamlit Community Cloud.

## Model

The final model is selected using the lowest RMSE from the four evaluated models. In the current notebook results, Linear Regression is the selected final model.
