# SpendSense AI — Render Cloud Deployment Guide (with MongoDB)

This guide walks you through deploying **SpendSense AI** to [Render](https://render.com) connected to a **MongoDB Atlas** cloud database.

---

## 📋 Prerequisites

1. A **GitHub account** ([github.com](https://github.com)).
2. A **Render account** ([render.com](https://render.com)).
3. A **free MongoDB Atlas database** ([mongodb.com/atlas](https://www.mongodb.com/cloud/atlas) — 100% free forever tier).

---

## 🍃 Step 1: Set up Free MongoDB Atlas Database (2 minutes)

1. Sign up or log into [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
2. Click **Create** &rarr; Select **M0 Free Cluster**.
3. Under **Security Quickstart**:
   - Create a database user (e.g., Username: `spendsense_user`, Password: `your-password`).
   - Under **Network Access**, click **Add IP Address** &rarr; Select **Allow Access from Anywhere** (`0.0.0.0/0`) so Render can connect.
4. Click **Connect** &rarr; Select **Drivers** (Python).
5. Copy your connection string:
   ```text
   mongodb+srv://spendsense_user:<password>@cluster0.xxxx.mongodb.net/spendsense_ai?retryWrites=true&w=majority
   ```
   *(Replace `<password>` with your database user password).*

---

## 🚀 Step 2: Push Your Project to GitHub

1. Open your terminal in this project folder:
   ```bash
   git add .
   git commit -m "feat: Connect MongoDB and simplify user registration flow"
   ```
2. If you haven't linked your GitHub repository yet:
   ```bash
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/spendsense-ai.git
   git push -u origin main
   ```
   If already linked:
   ```bash
   git push origin main
   ```

---

## ☁️ Step 3: Deploy on Render

### Method A: One-Click Render Blueprint (Recommended)

1. Open your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** &rarr; Select **Blueprint**.
3. Select your GitHub repository (`spendsense-ai`).
4. Under Environment Variables, paste your `MONGO_URI` from Step 1.
5. Click **Apply**. Render will automatically build and deploy!

### Method B: Manual Web Service

1. On [Render Dashboard](https://dashboard.render.com), click **New +** &rarr; **Web Service**.
2. Connect your GitHub repository.
3. Configure the following fields:
   - **Name**: `spendsense-ai`
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
4. Under **Environment Variables**, add:
   - `PYTHON_VERSION` = `3.12.6`
   - `SECRET_KEY` = *(click Generate or type a random string)*
   - `MONGO_URI` = *(Paste your MongoDB Atlas connection string from Step 1)*
   - `FLASK_DEBUG` = `0`
5. Click **Create Web Service**.

---

## ✅ Step 4: Verification

1. Once the deployment says **Live**, open your URL: `https://your-app-name.onrender.com`.
2. Click **Create Account** to register your own account.
3. Add a few transactions or upload a receipt to verify real-time AI categorization and live MongoDB persistence!
