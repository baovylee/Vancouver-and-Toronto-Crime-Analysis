# 🔍 Toronto vs Vancouver Crime Analysis

An end-to-end Data Science project comparing crime patterns in Toronto and Vancouver using official open-government data, exploratory data analysis, machine learning, and an interactive Streamlit dashboard.

🚀 **Live Demo:** https://vancouver-and-toronto-crime-analysis.streamlit.app/

📂 **Dataset:** 696,000+ crime incidents (2016–2025)

---

## 📸 Dashboard Preview



## 📌 Project Overview

This project analyzes over 696,000 crime incidents reported in Toronto and Vancouver between 2016 and 2025.

The objective is to identify differences in crime patterns, explore temporal and geographic trends, cluster neighbourhoods by risk level, and predict crime type using machine learning.

---

## 🔑 Key Findings

* Toronto is dominated by **Violent Crime (60.3%)**
* Vancouver is dominated by **Property Crime (76.6%)**
* Both cities experience peak crime activity around **midnight**
* Crime volume is highest during **summer months**
* Vancouver's Central Business District records over **100,000 incidents**
* Crime patterns changed significantly during the COVID-19 period

---

## 🤖 Machine Learning

### K-Means Clustering

Neighbourhoods were grouped into:

* High Risk
* Medium Risk
* Low Risk

Using:

* Crime volume
* Violent crime percentage
* Property crime percentage
* Average crime hour
* Night-time crime percentage

### Random Forest Classifier

**Goal:** Predict whether a crime is Property Crime or Violent Crime.

**Features:**

* City
* Neighbourhood
* Month
* Hour

### Results

| Metric    | Property Crime | Violent Crime |
| --------- | -------------- | ------------- |
| Precision | 0.99           | 0.60          |
| Recall    | 0.62           | 0.99          |
| F1 Score  | 0.76           | 0.75          |

### Overall Accuracy

**75%**

### Key Insight

The city where a crime occurs is the strongest predictor of crime type, contributing approximately **75% of total feature importance**.

---

## 📊 Dashboard Features

### Overview

* KPI summary cards
* Crime distribution analysis
* Neighbourhood risk ranking

### Trends

* Yearly crime trends
* Monthly seasonality
* Hourly crime heatmaps

### Crime Maps

* Interactive heatmaps
* Crime-type layer visualization
* Geographic hotspot exploration

### ML Predictor

* Crime type prediction
* Confidence score display
* Interactive user inputs

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Folium
* GeoPandas
* Scikit-Learn
* Streamlit

---

## 📁 Project Structure

```text
crime-analysis/
├── data/
├── notebooks/
├── output/
│   ├── figures/
│   ├── maps/
│   └── models/
├── app/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/baovylee/Vancouver-and-Toronto-Crime-Analysis

cd Vancouver-and-Toronto-Crime-Analysis

pip install -r requirements.txt
```

---

## 🚀 Run Locally

```bash
streamlit run app/app.py
```

---

## 📄 Data Sources

* Toronto Police Service Open Data
* Vancouver Police Department Open Data

All datasets are publicly available through official government open-data portals.

---

