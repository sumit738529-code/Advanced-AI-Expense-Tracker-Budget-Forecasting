# ✦ SpendSense AI — Expense Tracker & Budget Forecasting (MongoDB + AI)

A modern, intelligent personal finance web application built with **Flask**, **MongoDB**, and **Scikit-Learn Machine Learning**.

SpendSense AI automatically categorizes your transactions as you type, forecasts upcoming monthly spending, audits recurring subscriptions, and detects unusual expenditure spikes.

---

## 🌟 Key Features

1. **User Authentication & Personal Ledger**: Register an account, log in securely with salted password hashing, and manage your private financial records stored in MongoDB.
2. **AI Category Suggestions**: Scikit-Learn TF-IDF + Multinomial Naive Bayes engine suggests expense categories in real-time as you type descriptions.
3. **Time-Series Budget Forecasting**: Multi-model forecast combining Linear Trend extrapolation and Double Exponential Smoothing with 90% confidence bands.
4. **Outlier & Anomaly Detection**: Statistical Interquartile Range (IQR) and Z-score filter flags unusual spikes and duplicate charges.
5. **Smart Receipt OCR**: Scan receipt photos or paste invoice text to auto-fill merchant names, dates, and amounts into your ledger.
6. **50/30/20 Financial Health Score (0-100)**: Evaluates Needs vs. Wants vs. Savings and provides actionable budgeting tips.
7. **Recurring Subscriptions Manager**: Tracks ongoing subscriptions (Netflix, Spotify, Wifi, Gym) and calculates annual commitment.
8. **PDF & CSV Statements**: Download executive monthly financial reports (generated via ReportLab) and CSV transaction logs.
9. **AI Financial Copilot**: Floating conversational assistant providing contextual financial advice.

---

## 🍃 MongoDB Integration

SpendSense AI connects to MongoDB via `pymongo`:
- **Local MongoDB**: `mongodb://localhost:27017/spendsense_ai`
- **MongoDB Atlas Cloud**: `mongodb+srv://<username>:<password>@cluster0.xxxx.mongodb.net/spendsense_ai?retryWrites=true&w=majority`
- **In-Memory Fallback**: When running locally without a MongoDB server running, the app automatically initializes an in-memory database engine for seamless testing.

---

## 💻 Local Setup & Running

1. **Clone or open the folder**:
   ```powershell
   cd SpendSense_AI_Expense_Budget_Forecasting
   ```

2. **Activate your virtual environment**:
   ```powershell
   # If PowerShell execution policy restricts scripts, allow it for this process:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```powershell
   python app.py
   ```
   Open **http://127.0.0.1:5000** in your browser, click **Create Account**, and start adding your transactions!

5. **Run automated test suite**:
   ```powershell
   python -m pytest -v
   ```

---

## 🌐 Deploy to Render Cloud

See [DEPLOYMENT.md](DEPLOYMENT.md) for full step-by-step instructions.

Summary:
1. Push project to a GitHub repository.
2. Create a free cluster on [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) and copy the connection string.
3. On [Render](https://render.com), create a new **Web Service** or **Blueprint**, set `MONGO_URI` to your Atlas connection string, and deploy!
