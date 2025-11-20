# Social Media Trend & Engagement Analysis

## Overview
This project analyzes synthetic social media post data to identify topic trends, estimate engagement drivers, and build a predictive model for total engagement. It demonstrates natural language processing, feature extraction, and regression modeling.

---

## Objectives
- Generate realistic social media post data
- Build NLP feature representations using TF-IDF
- Predict total engagement (likes + comments + shares)
- Produce analytics aggregations suitable for dashboards

---

## Project Structure
```
social-media-trend-analysis/
│
├── data/
│   └── synthetic/                # Synthetic post dataset
│
├── src/
│   ├── data/
│   │   └── generate.py           # Post generator
│   └── models/
│       ├── train.py              # Vectorizer + regression model
│       └── evaluate.py           # Topic/time aggregates
│
├── artifacts/                    # Saved TF-IDF vectorizer + model
└── reports/
    └── aggregates/               # CSV files for dashboards
```

---

## Methods

### **1. Synthetic Data Generation**
Each post includes:
- User ID  
- Timestamp  
- Topic label  
- Text content  
- Engagement metrics  

Topics include AI, sports, gaming, finance, travel, food, etc.

### **2. Feature Engineering**
- TF-IDF vectorization  
- Bigram extraction  
- Sparse matrix modeling  

### **3. Engagement Prediction**
- Linear Regression baseline  
- Predicts:  
  ```
  total_engagement = likes + comments + shares
  ```

### **4. Reporting Outputs**
Generates:
- Average engagement by topic  
- Daily engagement over time  
- CSV files compatible with:
  - Tableau
  - Power BI
  - Any BI dashboard

---

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Generate Data
```bash
python src/data/generate.py
```

### Train Model
```bash
python src/models/train.py
```

### Produce Analytics Aggregates
```bash
python src/models/evaluate.py
```

---

## Key Results
- Identifies which topics generate highest average engagement  
- Baseline model provides a measurable predictor of engagement levels  
- Demonstrates end-to-end NLP → modeling → analytics workflow  

---

## Future Enhancements
- Add sentiment analysis (VADER/TextBlob)
- Perform topic modeling (LDA/BERT embeddings)
- Add boosted models (XGBoost/LightGBM)
- Build dashboard using Streamlit or Plotly

---

## License
MIT License

