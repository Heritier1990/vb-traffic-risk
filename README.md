# Virginia Beach Traffic and Accident Risk Intelligence

A Streamlit web application that predicts accident risk and traffic congestion for road segments in Virginia Beach, VA using machine learning and live weather data.

**Live App:** https://virginia-beach-traffic-risk.streamlit.app

---

## Research Question

Can an integrated machine learning and geospatial analytics system accurately predict accident risk and congestion hotspots across Virginia Beach road segments?

---

## Datasets

| Dataset | Source | Records |
|---|---|---|
| Traffic crash reports | Virginia Beach Police Department | 4,752 |
| Annual Average Daily Traffic | Virginia Department of Transportation (VDOT) | 1,200 |

---

## Model

Algorithm: XGBoost multiclass classifier  
Performance: Accident risk F1 score 0.46, Congestion risk F1 score 0.48

---

## How to Run Locally

1. Clone the repository: `git clone https://github.com/your-username/vb-traffic-risk.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys: `OPENWEATHER_API_KEY=your_key` and `GROQ_API_KEY=your_key`
4. Run the app: `streamlit run app.py`

---

## Tech Stack

Python, Streamlit, XGBoost, scikit-learn, pandas, folium, streamlit-folium, OpenWeatherMap API, Groq API

---

## Data Sources

Virginia Beach Police Department, Traffic Crash Reports 2023  
Virginia Department of Transportation (VDOT), Annual Average Daily Traffic counts 2023

---

## Author

Capstone Project, DASC 690, Virginia Beach Virginia, Training year 2023
