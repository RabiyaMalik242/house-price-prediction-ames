# 🏠 House Price Prediction — Ames Housing Dataset

A machine learning project that predicts house sale prices using the Ames Housing Dataset — a rich, real-world dataset with 1,460 houses and 79 features. Built as part of an AI/ML Internship.

---

## 📌 Objective

Build and evaluate regression models to predict house sale prices using property features such as overall quality, living area, neighborhood, garage, and age of the house.

---

## 📂 Dataset

- **Source:** [Ames Housing Dataset — Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- **File:** `AmesHousing.csv`
- **Size:** 1,460 houses, 79 features
- **Target:** `SalePrice` — Sale price of the house

**Key Features:**

| Feature | Description |
|---|---|
| `Overall Qual` | Overall material and finish quality (1–10) |
| `Gr Liv Area` | Above grade living area (sq ft) |
| `Garage Cars` | Garage capacity in car count |
| `Total Bsmt SF` | Total basement area (sq ft) |
| `Year Built` | Original construction year |
| `Neighborhood` | Physical location within Ames, Iowa |
| `Full Bath` | Full bathrooms above grade |
| `Bedroom AbvGr` | Bedrooms above ground |

---

## 🛠️ Tools & Libraries

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib

---

## ⚙️ Project Pipeline

1. **Data Loading & Inspection** — shape, dtypes, missing value analysis
2. **Exploratory Data Analysis (EDA)**
   - SalePrice distribution (raw + log + box plot)
   - Scatter plots vs top correlated features
   - SalePrice by Neighborhood
   - Overall Quality impact on price
   - Correlation heatmap (top 15 features)
   - Outlier detection
3. **Preprocessing**
   - Drop high-missing columns (>40% NaN)
   - Fill remaining missing values (median/mode)
   - Label encode all categorical columns
   - Feature engineering (house_age, total_sf, total_bathrooms, total_porch_sf)
   - Train/test split (80/20)
   - StandardScaler with before vs after comparison
4. **Model Training**
   - Linear Regression (log-transformed target)
   - Gradient Boosting Regressor (tuned, log-transformed target)
5. **Evaluation**
   - MAE, RMSE, R²
   - Actual vs Predicted scatter plots
   - Residual plots and residual distribution
   - Model comparison table and bar charts
6. **Feature Importance Analysis**
   - Linear Regression coefficients (top 15)
   - Gradient Boosting importance scores (top 15)
7. **Predictions**
   - Test set predictions with error % analysis
   - Sorted actual vs predicted line chart
   - Single house price estimate with visual
   - Batch predictions for 5 houses (varying quality/size)
8. **Model Saving** — joblib (.pkl)

---

## 📊 Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | \$17,504 | \$37,369 | 0.8258 |
| Gradient Boosting | \$13,300 | \$22,157 | 0.9388 |

> Best Model by R²: Gradient Boosting.

---

## 🔑 Key Findings

- `Overall Qual` is the single strongest predictor of house price in Ames
- `Gr Liv Area` (living area sq ft) is the second most important feature
- Neighborhood significantly impacts price — top neighborhoods command 2–3x premium over bottom ones
- Log-transforming the skewed SalePrice target significantly improves model fit
- Feature engineering (house_age, total_sf, total_bathrooms) added meaningful signals
- Gradient Boosting outperforms Linear Regression by capturing non-linear feature interactions

---

## 💾 Models

Run the notebook to train and save the models locally. The following files will be generated:

```
ames_lr_model.pkl
ames_gb_model.pkl
ames_scaler.pkl
```

> ⚠️ The scaler must be loaded alongside Linear Regression — new inputs must be scaled before prediction. Gradient Boosting uses raw features directly.

---

## 📁 Repository Structure

```
house-price-prediction-ames/
│
├── house_price_prediction_ames.ipynb   # Main notebook
├── AmesHousing.csv                     # Dataset
├── images/                             # Saved plot images
└── README.md
```

---

## 🚀 How to Run

1. Clone the repository
```bash
   git clone https://github.com/RabiyaMalik242/house-price-prediction-ames.git
```
2. Install dependencies
```bash
   pip install -r requirements.txt
```
3. Download `AmesHousing.csv` from [Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) and place it in the project folder
4. Open and run `house_price_prediction_ames.ipynb` top to bottom

---

## 📝 Note on Metrics

This is a **regression task** — classification metrics like accuracy, precision, and recall do not apply. The correct evaluation metrics are:

| Metric | What it measures |
|---|---|
| **MAE** | Average absolute difference between predicted and actual price |
| **RMSE** | Like MAE but penalizes large errors more heavily |
| **R²** | % of price variation explained by the model (closer to 1.0 = better) |

---

## 👩‍💻 Author

Rabiya Malik BS Software Engineering — AI/ML Internship Project
