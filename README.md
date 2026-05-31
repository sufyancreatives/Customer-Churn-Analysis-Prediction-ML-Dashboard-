# 📡 Telco Sentry — Advanced Churn Analytics Platform

Welcome to **Telco Sentry**, a premium, production-grade, multi-page machine learning dashboard designed to analyze subscriber behaviors, predict active churn risks, and provide actionable business intelligence to boost customer retention.

Designed with a high-fidelity **Neon Purple glassmorphic dark theme**, Telco Sentry transforms raw telecom data into an interactive visual cockpit.

---

## 🎨 Theme & Design Aesthetics
Telco Sentry is built from the ground up to offer an immersive visual experience:
- **Primary Theme**: Neon Purple (`#bd00ff`), Neon Pink (`#ff007f`), Neon Cyan (`#00f0ff`), and Electric Violet (`#8b5cf6`).
- **Layout Architecture**: Glassmorphic panels (`backdrop-filter: blur(12px)`), card hover translations, customized scroll bars, and gradient text accents.
- **Data Visualizations**: Customized interactive Plotly charts utilizing a unified color palette, detailed dark-mode hover tooltips, and glowing neon purple gridlines.

---

## ⚙️ Core Application Architecture (6 Screen Pipeline)

### 1. 🏠 Home & Business Intelligence Hub
*   **KPI Cards Dashboard**: Real-time business metrics including Total Customers Analysed, Overall Churn Rate, Average Monthly Revenue/Subscriber, and Average Tenure length.
*   **Financial Churn Impact Calculator**: An interactive business modeling slider. Input custom Annual Customer Values, Target Retention Rates, and Campaign Retention Costs to instantly calculate net saved revenue by deploying retention programs.

### 2. 📂 Dataset Explorer & Profiler
*   **Core Characteristics Panel**: Instantly displays Database Card counts, features list, and missing ratio badges.
*   **Real-time Database Query Grid**: Interactive data grid viewer supporting column filtration, records range sliders, and keyword search.
*   **Descriptive Statistics**: Detailed numerical distributions and category schemas.
*   **Distribution Profiler**: Dynamically switches layouts to render high-contrast histograms (numerical columns) or bar count charts (categorical columns).

### 3. 📊 Interactive EDA Dashboard
*   **Univariate Analysis**: Renders pie percentages or violin-density distributions based on selected features.
*   **Bivariate Analysis (vs Churn)**: Contrasts target features side-by-side using grouped frequency bars or split boxplots.
*   **Correlation Matrix Heatmap**: A custom Pearson correlation heatmap with color transitions from Neon Pink to deep space purple.
*   **Feature Importance Tab**: Displays top-12 ranked predictors extracted from the active champion model.

### 4. ⚙️ Model Training Factory
*   **Pipeline Settings**: Select core algorithms (Logistic Regression, Decision Trees, RandomForest, Support Vector Classifier, KNN, GradientBoost, XGBoost).
*   **Holdout Ratio Slider**: Customize validation train-test split splits.
*   **SMOTE Toggle**: Enable/disable SMOTE (Synthetic Minority Over-sampling Technique) to automatically balance skewed active churn classes.
*   **Interactive Confusion Matrix Heatmap**: Detailed diagnostics mapping true negatives, false negatives, true positives, and false positives.
*   **Champion Model Registration**: Click-to-register the newly trained pipeline on disk to instantly deploy it into production.

### 5. 🏆 Model Comparison & Benchmarks
*   **Performance Leaderboard**: Renders a comprehensive candidate leaderboard styled with a custom purple color gradient ranking F1 Score, Recall, Precision, Accuracy, and ROC-AUC.
*   **Dual Classifier Curves**: Interactive Plotly plots rendering receiver operating characteristic (ROC) curves and Precision-Recall (PR) curves side-by-side.
*   **Champion Badge**: Highlights the top benchmarked algorithm with gold gradient glows.

### 6. 🔮 Live Predictive Intelligence System
*   **Customer Profiler Form**: A sleek 3-column interactive form containing demographic inputs, subscription terms, connectivity configurations, and monthly billing sliders.
*   **Prediction Indicator Gauge**: Real-time Churn Risk Meter rendering risk levels:
    *   🟢 **Safe / Low Risk** (< 35%)
    *   🟡 **Medium Churn Warning** (35% - 65%)
    *   🔴 **High Churn Risk** (> 65%)
*   **Personalized Risk Drivers Breakdown**: Automatically extracts positive and negative risk factor tags (e.g. Month-to-month contracts, lack of premium tech support, autopay registration) to explain individual client probability metrics.

---

## 🛠️ Technology Stack & Dependencies
*   **Frontend**: Streamlit Custom Dark UI
*   **Graphics Engine**: Plotly Interactive charts
*   **Calculations**: Numpy, Pandas, Scipy
*   **ML Engine**: Scikit-Learn pipelines, XGBoost
*   **Class Balancing**: Imbalanced-Learn (SMOTE)
*   **Model Serialization**: Joblib

---

## 📁 Repository Directory Structure
```text
├── app.py                # Main multi-page Streamlit dashboard router & layouts
├── utils.py              # Data cleaning, pipeline compilers, and storage utils
├── requirements.txt      # Production package dependencies
├── models/
│   └── best_churn_model.pkl   # Exported scikit-learn champion pipeline
└── notebook/
    └── eda_&_preprocesing.py  # Exploratory notebook/script for model validation
```

---

## 🚀 Setup & Execution Guide

### 1. Environment Preparation
Ensure you are using Python 3.10+ (tested on Python 3.13.x).

```bash
# Clone the repository
git clone <repository_url>
cd telco-churn-analytics

# Install active package requirements
pip install -r requirements.txt
```

### 2. Run the Dashboard
To start the server and run the platform dashboard locally:

```bash
streamlit run app.py
```

The application will launch automatically in your default browser at: `http://localhost:8501/`

### 3. Model Training & Live Inference
1. Navigate to the **⚙️ Model Factory** page.
2. Select an algorithm (e.g., **XGBoost** or **RandomForest**).
3. Hit **Train Classifier Pipeline**.
4. Click **Register as Live Champion Model**.
5. Go to the **🔮 Prediction System** tab and start predicting customer churn rates interactively!

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.
