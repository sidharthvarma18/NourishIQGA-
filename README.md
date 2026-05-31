# 🥗 NourishIQ – Business Data Analytics Dashboard

**Personalised Diet, Nutrition & Fitness App for India**

---

## 📌 Project Overview

NourishIQ is a full-stack Business Data Analytics project built for academic assessment. It demonstrates the complete data science and analytics lifecycle applied to a real-world HealthTech business idea targeting the Indian market.

The dashboard is built with **Streamlit** and covers all major BDA concepts including descriptive analytics, classification, decision trees, clustering, association rule mining, regression, forecasting, recommender system, text mining, and social network analysis.

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset (if CSV is missing)

```bash
python generate_dataset.py
```

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

### 4. Open in Browser

Visit: `http://localhost:8501`

---

## 📁 Project Structure

```
nourishiq/
├── app.py                   # Main Streamlit dashboard (15 sections)
├── generate_dataset.py      # Synthetic dataset generator
├── nourishiq_dataset.csv    # 2,000-row synthetic survey dataset (80 variables)
├── requirements.txt         # Python package dependencies
└── README.md                # This file
```

---

## 📊 Dashboard Sections

| Section | Analytics Type | Key Technique |
|---|---|---|
| Home & Objectives | Overview | Business framing |
| Descriptive Analytics | Descriptive | KPIs, charts, distributions |
| Diagnostic Analytics | Diagnostic | Correlations, cross-tabs |
| Classification Models | Predictive | Random Forest, Logistic Reg, GBM |
| Decision Tree Analysis | Predictive | Decision tree rules |
| Customer Segmentation | Predictive | K-Means clustering |
| Association Rule Mining | Predictive | Apriori algorithm |
| Regression Analysis | Predictive | Random Forest Regressor |
| Forecasting | Predictive | Moving Average, Linear Trend |
| Recommender System | Prescriptive | Rule-based personalisation |
| Text Mining & Sentiment | Predictive | VADER Sentiment, WordCloud |
| Social Network Analysis | Descriptive | NetworkX referral graph |
| Ethics & Data Privacy | Governance | DPDP Act, AI governance |
| Business Recommendations | Prescriptive | Data-driven action plan |
| Predict New User | Predictive | Real-time ML prediction |

---

## 🗃️ Dataset Description

- **2,000 synthetic Indian respondents**
- **80 variables** covering demographics, health, diet, fitness, psychographics, and spending
- **15 engineered features** including Health Risk Score, Subscription Potential, Customer Value Score
- **Target variables**: Sign-Up Intent (Low/Medium/High), Likely to Subscribe (Yes/No), Monthly Budget

---

## 🛠️ Technology Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| Streamlit | Dashboard framework |
| Pandas / NumPy | Data processing |
| Scikit-learn | Classification, Regression, Clustering |
| Plotly | Interactive visualisations |
| MLxtend | Association Rule Mining (Apriori) |
| NetworkX | Social network / referral analysis |
| WordCloud + VADER | Text mining & sentiment analysis |
| Matplotlib | Word cloud rendering |

---

## 🌐 Streamlit Cloud Deployment

1. Push this folder to a GitHub repository (no sub-folders needed)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path**: `app.py`
5. Click Deploy

---

## ⚠️ Disclaimer

This project uses a **synthetic dataset generated for educational purposes**. All respondent data is simulated. NourishIQ is an academic business idea and not a live commercial product. Health recommendations in this dashboard are illustrative only and do not replace professional medical advice.

---

## 👨‍💻 Author

NourishIQ Analytics Team – Business Data Analytics Assessment Project  
Dataset: 2,000 synthetic Indian respondents | 80 variables | 15 sections
