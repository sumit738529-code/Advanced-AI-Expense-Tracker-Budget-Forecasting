# SpendSense AI — Render Cloud Deployment Guide

This guide walks you through deploying **SpendSense AI** to [Render](https://render.com) as a production web application with zero friction.

---

## 📋 Prerequisites

1. A **GitHub account** ([github.com](https://github.com)).
2. A **Render account** ([render.com](https://render.com) — free tier available).

---

## 🚀 Method 1: One-Click Render Blueprint (Recommended)

SpendSense AI includes a pre-configured `render.yaml` Blueprint file.

### Step 1: Push Project to GitHub

1. Open your terminal or Command Prompt in this project folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: SpendSense AI advanced project"
   ```
2. Create a new repository on GitHub (e.g., `spendsense-ai`).
3. Link and push your repository:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/spendsense-ai.git
   git push -u origin main
   ```

### Step 2: Deploy on Render via Blueprint

1. Go to your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** &rarr; Select **Blueprint**.
3. Connect your GitHub repository (`spendsense-ai`).
4. Render will automatically detect `render.yaml` and configure:
   - **Service Name**: `spendsense-ai`
   - **Runtime**: `Python 3.12.6`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --log-file -`
   - **Auto-generated SECRET_KEY**
5. Click **Apply**. Render will build and deploy your application within 2-3 minutes!
6. Once deployed, Render will provide a live HTTPS URL (e.g., `https://spendsense-ai.onrender.com`).

---

## 🛠️ Method 2: Manual Web Service Setup on Render

If you prefer to configure the Web Service manually:

1. Log in to [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** &rarr; **Web Service**.
3. Select **Build and deploy from a Git repository** and connect your repository.
4. Fill in the following details:
   - **Name**: `spendsense-ai` (or your preferred name)
   - **Region**: Choose the closest region (e.g., Singapore, Frankfurt, Oregon)
   - **Branch**: `main`
   - **Root Directory**: Leave blank (root of project)
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     gunicorn app:app --log-file -
     ```
   - **Instance Type**: `Free`
5. Scroll down to **Environment Variables** and add:
   - `PYTHON_VERSION` = `3.12.6`
   - `SECRET_KEY` = *(click "Generate" or enter a long random string)*
   - `FLASK_DEBUG` = `0`
   - *(Optional)* `GEMINI_API_KEY` = *(Your Google Gemini API key if using generative Copilot)*
6. Click **Create Web Service**.

---

## 🗄️ Optional: Connecting a Free Render PostgreSQL Database

By default, SpendSense AI runs with zero setup using an SQLite database file. If you want a persistent PostgreSQL database on Render:

1. In Render Dashboard, click **New +** &rarr; **PostgreSQL**.
2. Name it `spendsense-db` and select the **Free** tier.
3. Once created, copy the **Internal Database URL** (e.g., `postgres://user:pass@host/db`).
4. Open your `spendsense-ai` Web Service &rarr; **Environment** tab.
5. Add/Edit the variable:
   - `DATABASE_URL` = `<PASTE_INTERNAL_DATABASE_URL>`
6. SpendSense AI automatically converts `postgres://` to `postgresql://` and creates all tables on startup!

---

## ✅ Post-Deployment Verification Checklist

Once the build finishes:

1. Open your live application URL: `https://your-app-name.onrender.com`
2. Click **Viva Demo Login** to verify pre-seeded data, graphs, and health scores load.
3. Visit the **AI Model Lab** (`/viva-lab`) to inspect the Confusion Matrix and live NLP Sandbox.
4. Test the **Smart Receipt OCR** (`/receipt-scanner`) with the sample bill button.
5. Export your first **Monthly Statement (PDF)** (`/export/pdf`).

---

## 🎓 Viva Presentation Tips for Professors & Evaluators

| Feature | Viva Talking Point |
|---|---|
| **Text Categorization** | Explain the **TF-IDF + Multinomial Naive Bayes** pipeline with Laplace smoothing. Show token weights and class distribution. |
| **Spending Forecast** | Highlight the **Multi-Model Ensemble** (Holt's Double Exponential Smoothing + OLS Trend + Weighted Moving Average) and 90% confidence bands. |
| **Anomaly Detection** | Explain the **IQR ($Q_3 + 1.75 \times \text{IQR}$) and Z-score ($Z > 2.0$)** outlier thresholding for spending spikes and POS duplicate charge detection. |
| **Financial Health** | Present the **50/30/20 rule** algorithmic implementation, scoring components (0-100), and personalized grading. |
| **Receipt Scanner** | Demonstrate Computer Vision regex document parsing and auto-population of ledger fields. |
