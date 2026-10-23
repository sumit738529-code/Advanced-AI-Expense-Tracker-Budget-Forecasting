# ✦ SpendSense AI — Autonomous Expense Categorization & Budget Forecasting

An MSc Computer Applications / B.Tech Computer Science Final-Year Capstone Project.

SpendSense AI is an intelligent personal finance platform engineered with **Scikit-Learn NLP text classification**, **multi-model time-series forecasting**, **statistical anomaly detection**, **smart receipt OCR parsing**, and **50/30/20 financial health analytics**.

---

## 🌟 Advanced Project Features

### 1. 🧠 Machine Learning NLP Categorizer (Scikit-Learn)
- **Architecture**: Sublinear TF-IDF Vectorizer with unigrams & bigrams coupled to Laplace-smoothed Multinomial Naive Bayes.
- **Explainability**: Live token-level feature attribution showing why a specific transaction was assigned to a category.
- **Dynamic Keystroke Inference**: REST endpoint `/api/categorize` triggers real-time prediction badges as users type descriptions.

### 2. 📈 Multi-Model Time-Series Spending Forecast
- **Ensemble Engine**: Combines **Ordinary Least Squares (OLS) Linear Trend** (35%), **Holt's Double Exponential Smoothing** (45%), and **3-Month Weighted Moving Average** (20%).
- **Confidence Bands**: Generates parametric 90% confidence interval upper and lower prediction boundaries.
- **Backtested Metrics**: Calculates historical Mean Absolute Error (MAE) and trend trajectory.

### 3. 🚨 Spending Anomaly & Duplicate Charge Detection
- **Statistical Filtering**: Utilizes an Interquartile Range ($Q_3 + 1.75 \times \text{IQR}$) and $Z$-score ($Z > 2.0$) hybrid filter.
- **Spike Recognition**: Automatically flags unusual expenditure surges against the user's 90-day rolling category baseline.
- **Duplicate Protection**: Detects identical amounts logged within a 48-hour window.

### 4. 🧾 Smart Receipt & Bill OCR Parser
- **Computer Vision & Heuristics**: Extracts merchant name, billing date, line items, and total amount from receipt photos, invoices, and raw text bills.
- **1-Click Ledger Conversion**: Allows instantaneous verification and transfer into the user's verified expense history.

### 5. 🎯 50/30/20 Financial Health Score (0 - 100)
- **Algorithmic Grading**: Automatically classifies expenses into **Needs** (Housing, Food, Utilities, Healthcare), **Wants** (Shopping, Entertainment, Travel), and **Savings** (Net Surplus).
- Computes an objective credit-style wellness score (0 to 100) and letter grades (A+, A, B, C, D) with personalized budget advice.

### 6. 🔁 Subscription & Recurring Outflow Manager
- **Cadence Scanner**: Analyzes historical ledger cycles to automatically detect recurring commitments (Netflix, Spotify, Wifi, Gym, Amazon Prime).
- **Burden Projection**: Calculates normalized monthly recurring commitment and total annual burden.

### 7. 🔬 AI Model & Viva Defense Lab (`/viva-lab`)
- Dedicated evaluator dashboard displaying:
  - Overall Accuracy, Macro Precision, Macro Recall, and Macro F1-Score.
  - Multi-Class Confusion Matrix across all 11 financial categories.
  - Interactive Live Playground for examiners to test arbitrary sentences.
  - Time-series algorithmic comparison table.

### 8. 📄 Bank-Grade PDF & CSV Statements
- In-memory PDF generator using **ReportLab** producing formal monthly financial reports with summary cards, 50/30/20 tables, and transaction logs.
- Full CSV export and schema-validated bulk CSV import.

### 9. 🤖 AI Financial Copilot
- Context-aware conversational financial advisor analyzing user metrics in real-time.
- Works 100% offline via rule engine, with optional **Google Gemini API** integration if configured.

### 10. ⚡ 1-Click Viva Demo Data Seeder
- Instant population of 6 months of realistic historical transactions, budgets, savings goals, and deliberate anomaly spikes for viva presentation.

---

## 🏗️ Technology Stack

- **Backend**: Python 3.12, Flask, Flask-SQLAlchemy, Werkzeug
- **Machine Learning & Analytics**: Scikit-Learn (TF-IDF, Naive Bayes), NumPy, SciPy, Pandas
- **Document & PDF Generation**: ReportLab, Pillow (PIL), Pytesseract
- **Database**: SQLite (Local Dev) / PostgreSQL (Cloud Production)
- **Frontend**: HTML5, Bootstrap 5, Bootstrap Icons, Chart.js (CDN)
- **Production Server**: Gunicorn WSGI
- **Deployment**: Render Cloud Web Service (`render.yaml` Blueprint)

---

## 🚀 Quick Start (Local Setup)

### 1. Clone & Setup Virtual Environment

```bash
# Clone or navigate to the project directory
cd SpendSense_AI_Expense_Budget_Forecasting

# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1
# On macOS/Linux:
# source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the Application

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

- Click **Viva Demo Login** to instantly explore with 6 months of pre-analyzed demo data!

---

## 🧪 Run Automated Tests

SpendSense AI includes unit and regression tests for its ML classifier, evaluation metrics, and receipt parser:

```bash
python -m pytest -v
```

---

## 🌐 Deploy to Render Cloud

Detailed step-by-step instructions with screenshots and database setup are available in [DEPLOYMENT.md](DEPLOYMENT.md).

Quick summary:
1. Push project to a GitHub repository.
2. In [Render Dashboard](https://dashboard.render.com), click **New +** &rarr; **Blueprint**.
3. Select your repository. Render automatically reads `render.yaml` and starts your live web service with HTTPS!

---

## 📁 Project Architecture

```text
SpendSense_AI_Expense_Budget_Forecasting/
├── app.py                  # Core Flask application, routing, REST APIs
├── ai_engine.py            # ML classifier, time-series forecast, anomaly detection
├── models.py               # SQLAlchemy database models (User, Tx, Budget, Goal, Sub)
├── pdf_generator.py        # Executive PDF financial statement generator (ReportLab)
├── seed_data.py            # 6-Month realistic viva demo data seeder
├── requirements.txt        # Production dependencies
├── Procfile                # Gunicorn process definition for Render
├── runtime.txt             # Python 3.12.6 runtime declaration
├── render.yaml             # Render Blueprint configuration
├── DEPLOYMENT.md           # Step-by-step Render cloud deployment guide
├── README.md               # Project documentation
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules
├── tests/
│   └── test_ai_engine.py   # Unit tests for ML models & parser
├── templates/
│   ├── base.html           # Layout shell with navbar & AI Copilot drawer
│   ├── landing.html        # Modern fintech landing page
│   ├── login.html          # Authentication with 1-click Viva Demo button
│   ├── register.html       # User sign up with currency selection
│   ├── dashboard.html      # KPI metrics, forecast & category charts, health score
│   ├── transactions.html   # Ledger with filters, anomaly tags, and pagination
│   ├── transaction_form.html # Add/Edit form with live AI keystroke badge
│   ├── receipt_scanner.html # Smart Receipt OCR & document parser
│   ├── budget.html         # Monthly budget & category-wise burndown caps
│   ├── goals.html          # Savings milestones with progress trackers
│   ├── subscriptions.html  # Recurring subscriptions & annual burden
│   ├── viva_lab.html       # AI Model inspection, Confusion Matrix, metrics
│   ├── settings.html       # Profile preferences, currency, and diagnostics
│   └── import.html         # Bulk CSV upload with schema guide
└── static/
    └── style.css           # Custom styles, gradients, and responsive components
```

---

## 🎓 Viva Defense FAQ

- **Q: How does the AI categorize expenses without an external API?**  
  *A: It uses a Scikit-Learn TF-IDF vectorizer (with unigram and bigram token features) and a Multinomial Naive Bayes classifier trained on verified real-world transaction descriptions. It is 100% offline, explainable, and returns normalized confidence scores.*

- **Q: What happens if an expense is an anomaly?**  
  *A: The anomaly engine calculates rolling category medians and uses an Interquartile Range ($Q_3 + 1.75 \times \text{IQR}$) and $Z$-score filter. When a spike or duplicate occurs, it is flagged in the database and surfaced with an alert badge explaining the ratio above normal.*

- **Q: How does the forecasting algorithm compare against simple averages?**  
  *A: The system implements a comparative ensemble: OLS Linear Regression, Holt's Double Exponential Smoothing, and 3-Month Weighted Moving Average. The Viva Lab (`/viva-lab`) shows the exact predictions of each model and the ensemble 90% confidence bands.*
