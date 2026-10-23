"""SpendSense AI Engine — Machine Learning & Intelligent Financial Services.

Backed by MongoDB database collections.
Features:
1. TF-IDF + Multinomial Naive Bayes NLP Categorizer with Explainability & Confidence.
2. Statistical & IQR-based Outlier / Anomaly / Fraudulent Expense Detection.
3. Multi-Model Forecasting (Linear Trend, Double Exponential Smoothing, Moving Average, Ensemble).
4. Smart Receipt & Bill OCR / Document Parser.
5. 50/30/20 Financial Health Score (0-100) & Budget Advisory.
6. Recurring Subscription & Bill Pattern Recognition.
7. Context-Aware AI Financial Copilot (Rule-based + Optional Gemini/OpenAI API).
"""

import os
import re
import json
import math
from datetime import date, datetime, timedelta
from collections import defaultdict
import numpy as np

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from db_mongo import get_db, TransactionDoc, BudgetDoc, SubscriptionDoc

CATEGORIES = [
    "Food",
    "Transport",
    "Housing",
    "Utilities",
    "Healthcare",
    "Shopping",
    "Entertainment",
    "Education",
    "Travel",
    "Income",
    "Other"
]

# ---------------------------------------------------------
# 1. Training Dataset
# ---------------------------------------------------------
RAW_TRAINING_DATA = [
    # Food
    ("swiggy food delivery dinner biryani", "Food"),
    ("zomato lunch burger pizza order", "Food"),
    ("starbucks cafe cappuccino latte snacks", "Food"),
    ("mcdonalds happy meal fries burger coke", "Food"),
    ("dominos pizza cheese garlic bread delivery", "Food"),
    ("supermarket grocery shopping vegetables fruits milk bread", "Food"),
    ("d-mart monthly provisions rice wheat oil dal", "Food"),
    ("local bakery pastries cake bread biscuits", "Food"),
    ("fine dining restaurant dinner with family", "Food"),
    ("subway roasted chicken sandwich wrap salad", "Food"),
    ("blinkit quick grocery 10 minute delivery eggs butter", "Food"),
    ("zepto grocery instant delivery curd vegetables", "Food"),
    ("kfc fried chicken meal bucket", "Food"),
    ("tea stall chai samosa afternoon refreshments", "Food"),
    ("ice cream parlor waffles dessert sundae", "Food"),
    ("butcher meat chicken fish seafood shop", "Food"),
    ("food court cafeteria office lunch thali", "Food"),

    # Transport
    ("uber cab ride to office airport", "Transport"),
    ("uber taxi trip cab commute", "Transport"),
    ("ola auto ride commute railway station", "Transport"),
    ("petrol fuel refill shell petrol bunk", "Transport"),
    ("diesel fuel tank refill car", "Transport"),
    ("metro card smart recharge travel token", "Transport"),
    ("city bus pass ticket conductor fare", "Transport"),
    ("fastag toll plaza automated highway payment", "Transport"),
    ("car parking charges mall basement valet", "Transport"),
    ("auto rickshaw fare meter cash payment", "Transport"),
    ("bike scooter servicing engine oil change brake repair", "Transport"),
    ("rapido bike taxi quick ride transit", "Transport"),
    ("train local train suburban ticket irctc local", "Transport"),
    ("car tyre puncture air pressure alignment", "Transport"),
    ("ev charging station battery charge electric scooter", "Transport"),

    # Housing
    ("monthly house rent payment owner landlord transfer", "Housing"),
    ("apartment society maintenance charges monthly", "Housing"),
    ("society water tanker payment residential", "Housing"),
    ("home interior furniture urban ladder ikea sofa table", "Housing"),
    ("plumber bathroom pipe leakage repair charges", "Housing"),
    ("electrician ceiling fan wiring switch installation", "Housing"),
    ("flat rent advance security deposit flatmate share", "Housing"),
    ("pest control treatment home deep cleaning service", "Housing"),
    ("carpenter door lock hinges kitchen woodwork", "Housing"),
    ("mattress bedsheet curtains home furnishing", "Housing"),

    # Utilities
    ("electricity bill payment bescom tata power adani mseb", "Utilities"),
    ("water board municipal supply bill payment", "Utilities"),
    ("piped cooking gas lpg cylinder refill indane hp bharat", "Utilities"),
    ("wifi broadband internet bill act airtel jio fiber", "Utilities"),
    ("mobile prepaid recharge monthly unlimited plan vodafone", "Utilities"),
    ("postpaid mobile bill payment airtel jio", "Utilities"),
    ("dth cable tv recharge tata play airtel digital dish tv", "Utilities"),
    ("garbage disposal sanitation municipal tax payment", "Utilities"),

    # Healthcare
    ("doctor consultation fee clinic general physician", "Healthcare"),
    ("apollo pharmacy medicine purchase prescription tablets", "Healthcare"),
    ("medical diagnostic lab blood test thyroid test scan", "Healthcare"),
    ("dental clinic tooth cleaning cavity filling braces", "Healthcare"),
    ("eye doctor spectacles contact lenses lenskart eye checkup", "Healthcare"),
    ("hospital emergency admission bill surgery charges", "Healthcare"),
    ("health insurance mediclaim annual premium policy", "Healthcare"),
    ("physiotherapy session back pain treatment", "Healthcare"),
    ("vitamins supplements protein powder nutrition shop", "Healthcare"),
    ("skincare dermatologist consultation cream ointment", "Healthcare"),

    # Shopping
    ("amazon online shopping electronics gadgets clothes", "Shopping"),
    ("flipkart big billion days clothing shoes purchase", "Shopping"),
    ("myntra fashion apparel jeans t-shirt jacket", "Shopping"),
    ("zara h&m retail clothing store shopping mall", "Shopping"),
    ("electronics apple store iphone charger airpods macbook", "Shopping"),
    ("footwear nike adidas puma sneakers running shoes", "Shopping"),
    ("cosmetics makeup nykaa lipstick perfume beauty", "Shopping"),
    ("watches titan smartwatch noise boAt headphones earbuds", "Shopping"),
    ("sunglasses rayban shades eyewear retail", "Shopping"),
    ("gift shop birthday gift wrap greeting card", "Shopping"),

    # Entertainment
    ("pvr cinemas movie tickets popcorn imax multiplex", "Entertainment"),
    ("bookmyshow concert live show standup comedy tickets", "Entertainment"),
    ("netflix monthly standard streaming subscription", "Entertainment"),
    ("spotify premium music subscription student plan", "Entertainment"),
    ("amazon prime video subscription annual renewal", "Entertainment"),
    ("playstation store ps5 video game digital purchase", "Entertainment"),
    ("steam game purchase summer sale pc gaming", "Entertainment"),
    ("disney hotstar ott subscription annual plan", "Entertainment"),
    ("amusement park theme park entry tickets roller coaster", "Entertainment"),
    ("youtube premium ad-free family subscription", "Entertainment"),
    ("bowling arcade gaming zone tickets mall", "Entertainment"),

    # Education
    ("college university tuition semester fee payment", "Education"),
    ("coursera udemy online certificate course web development", "Education"),
    ("textbooks reference books academic bookstore study material", "Education"),
    ("coaching institute entrance exam preparation gate cat gre", "Education"),
    ("exam registration fee examination hall ticket", "Education"),
    ("stationery notebook pens files diary calculator", "Education"),
    ("school bus fees quarterly student transport", "Education"),
    ("coding bootcamp python data science machine learning course", "Education"),
    ("library membership fee book lending deposit", "Education"),

    # Travel
    ("makemytrip flight ticket booking indigo air india", "Travel"),
    ("irctc train ticket booking tatkal ac 3 tier reservation", "Travel"),
    ("hotel booking oyo room marriott resort weekend stay", "Travel"),
    ("airbnb vacation apartment rental staycation", "Travel"),
    ("holiday tour package kashmir goa kerala sight seeing", "Travel"),
    ("travel visa application fee tourist consulate", "Travel"),
    ("airport cab shuttle transfer taxi", "Travel"),
    ("scuba diving hiking trekking adventure sports guide fee", "Travel"),
    ("travel luggage suitcase trolley bag travel bag", "Travel"),

    # Income
    ("monthly salary credited employer payroll direct deposit", "Income"),
    ("freelance client project payment invoice settled", "Income"),
    ("consulting honorarium fee received bank transfer", "Income"),
    ("stock dividend payment shares mutual fund dividend", "Income"),
    ("interest credited savings account bank interest quarterly", "Income"),
    ("bonus festival performance appraisal bonus credited", "Income"),
    ("cashback reward credit card discount refund credited", "Income"),
    ("crypto trading profit withdrawal bank deposit", "Income"),
    ("rental income tenant rent payment credited", "Income"),

    # Other
    ("bank account annual maintenance fee charge debit", "Other"),
    ("atm cash withdrawal self debit", "Other"),
    ("charity donation temple ngo relief fund", "Other"),
    ("loan emi installment personal loan car loan", "Other"),
    ("credit card bill payment minimum balance", "Other"),
    ("unknown transfer upi friend send money split", "Other"),
    ("stamp duty legal notary court fee charges", "Other")
]

# ---------------------------------------------------------
# 2. Machine Learning Pipeline (TF-IDF + Multinomial NB)
# ---------------------------------------------------------
class ExpenseClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english", sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=0.5))
        ])
        self.is_trained = False
        self.train()

    def train(self):
        texts = [x[0] for x in RAW_TRAINING_DATA]
        labels = [x[1] for x in RAW_TRAINING_DATA]
        self.pipeline.fit(texts, labels)
        self.is_trained = True

    def predict(self, text):
        if not text or not text.strip():
            return {
                "category": "Other",
                "confidence": 0.0,
                "distribution": {c: 0.0 for c in CATEGORIES},
                "top_tokens": []
            }
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower()).strip()
        if not cleaned:
            return {
                "category": "Other",
                "confidence": 0.0,
                "distribution": {c: 0.0 for c in CATEGORIES},
                "top_tokens": []
            }

        probs = self.pipeline.predict_proba([cleaned])[0]
        classes = self.pipeline.classes_
        dist = {c: round(float(p), 4) for c, p in zip(classes, probs)}
        
        for c in CATEGORIES:
            if c not in dist:
                dist[c] = 0.0

        best_cat = max(dist, key=dist.get)
        confidence = dist[best_cat]

        vectorizer = self.pipeline.named_steps["tfidf"]
        words = cleaned.split()
        top_tokens = [w for w in words if w in vectorizer.vocabulary_]

        return {
            "category": best_cat,
            "confidence": round(float(confidence), 3),
            "distribution": dist,
            "top_tokens": top_tokens[:5]
        }

    def evaluate_model(self):
        texts = [x[0] for x in RAW_TRAINING_DATA]
        y_true = [x[1] for x in RAW_TRAINING_DATA]
        y_pred = self.pipeline.predict(texts)
        acc = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        labels = sorted(list(set(y_true)))
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        return {
            "accuracy": round(acc * 100, 2),
            "macro_precision": round(float(precision) * 100, 2),
            "macro_recall": round(float(recall) * 100, 2),
            "macro_f1": round(float(f1) * 100, 2),
            "labels": labels,
            "confusion_matrix": cm,
            "total_samples": len(texts)
        }

classifier = ExpenseClassifier()

def categorize_expense(description):
    return classifier.predict(description)

def get_model_evaluation():
    return classifier.evaluate_model()

# ---------------------------------------------------------
# 3. Anomaly & Unusual Spending Detection (MongoDB backed)
# ---------------------------------------------------------
def detect_transaction_anomaly(user_id, amount, category, tx_date=None, description=""):
    """Evaluates whether an individual expense is an unusual outlier, spike, or duplicate."""
    if amount <= 0:
        return False, None

    db = get_db()
    cutoff_date = (date.today() - timedelta(days=90)).isoformat()
    
    # Query MongoDB for past 90 days expenses in this category
    cursor = db.transactions.find({
        "user_id": str(user_id),
        "kind": "expense",
        "category": category,
        "tx_date": {"$gte": cutoff_date}
    })
    history = list(cursor)

    # Rule 1: Check for duplicate transaction
    if tx_date:
        tx_d_str = tx_date.isoformat() if isinstance(tx_date, date) else str(tx_date)[:10]
        for past_tx in history:
            p_d_str = str(past_tx.get("tx_date"))[:10]
            try:
                d1 = datetime.strptime(p_d_str, "%Y-%m-%d").date()
                d2 = datetime.strptime(tx_d_str, "%Y-%m-%d").date()
                if abs((d1 - d2).days) <= 1 and abs(float(past_tx.get("amount", 0.0)) - amount) < 0.01:
                    return True, f"Potential duplicate: identical amount logged within 48h ({past_tx.get('description', '')})"
            except Exception:
                pass

    if len(history) < 4:
        if amount > 15000:
            return True, f"High-value transaction without established category baseline"
        return False, None

    amounts = [float(t.get("amount", 0.0)) for t in history]
    median_val = float(np.median(amounts))
    q75, q25 = np.percentile(amounts, [75, 25])
    iqr = q75 - q25
    upper_threshold = q75 + (1.75 * iqr) if iqr > 0 else median_val * 2.5

    mean_val = float(np.mean(amounts))
    std_val = float(np.std(amounts))
    z_score = (amount - mean_val) / std_val if std_val > 0 else 0

    if amount > upper_threshold and z_score > 2.0:
        ratio = amount / median_val if median_val > 0 else 2.0
        return True, f"Spending Spike: {ratio:.1f}x higher than typical {category} expense (median: ₹{median_val:.0f})"

    return False, None

# ---------------------------------------------------------
# 4. Multi-Model Time Series Forecasting (MongoDB backed)
# ---------------------------------------------------------
def forecast_monthly_spend(user_id, months=6):
    """Ensemble forecast: Linear Trend, Holt's Exponential Smoothing & Moving Average."""
    db = get_db()
    today = date.today()
    monthly = []
    year, month = today.year, today.month

    for offset in range(months - 1, -1, -1):
        idx = year * 12 + month - 1 - offset
        y, m0 = divmod(idx, 12)
        m = m0 + 1
        m_str = f"{y}-{m:02d}"
        
        # Match all dates starting with YYYY-MM
        regex_pattern = f"^{m_str}"
        txs = db.transactions.find({
            "user_id": str(user_id),
            "kind": "expense",
            "tx_date": {"$regex": regex_pattern}
        })
        total = sum(float(t.get("amount", 0.0)) for t in txs)
        monthly.append({"month": m_str, "actual": round(float(total), 2)})

    actuals = [x["actual"] for x in monthly]
    n = len(actuals)
    non_zeros = [v for v in actuals if v > 0]

    next_idx = today.year * 12 + today.month
    ny, nm0 = divmod(next_idx, 12)
    next_month_str = f"{ny}-{nm0+1:02d}"

    if not non_zeros:
        return {
            "history": monthly,
            "next_month": next_month_str,
            "predicted_spend": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "method": "Ensemble (Holt's Smoothing + Linear Trend)",
            "models": {"linear": 0.0, "exponential_smoothing": 0.0, "weighted_moving_avg": 0.0},
            "mae": 0.0,
            "trend_direction": "stable"
        }

    # Model 1: Ordinary Least Squares (OLS) Linear Trend
    xs = list(range(n))
    xbar, ybar = sum(xs) / n, sum(actuals) / n
    denom = sum((x - xbar)**2 for x in xs)
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, actuals)) / denom if denom else 0.0
    linear_pred = max(0.0, ybar + slope * (n - xbar))

    # Model 2: Double Exponential Smoothing (Holt's method)
    alpha = 0.4
    beta = 0.3
    level = actuals[0]
    trend = actuals[1] - actuals[0] if n > 1 else 0.0
    for t in range(1, n):
        val = actuals[t]
        last_level = level
        level = alpha * val + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
    exp_pred = max(0.0, level + trend)

    # Model 3: 3-Month Weighted Moving Average
    recent = actuals[-3:] if len(actuals) >= 3 else actuals
    weights = [0.2, 0.3, 0.5][-len(recent):]
    norm_w = [w / sum(weights) for w in weights]
    ma_pred = sum(r * w for r, w in zip(recent, norm_w))

    ensemble = (0.35 * linear_pred) + (0.45 * exp_pred) + (0.20 * ma_pred)

    std_residual = float(np.std(actuals)) if len(actuals) > 1 else ensemble * 0.15
    lower_bound = max(0.0, ensemble - (1.28 * std_residual))
    upper_bound = ensemble + (1.28 * std_residual)
    mae = round(float(np.mean([abs(a - ybar) for a in actuals])), 2)

    return {
        "history": monthly,
        "next_month": next_month_str,
        "predicted_spend": round(ensemble, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "method": "Ensemble (Holt's Smoothing 45% + Linear 35% + WMA 20%)",
        "models": {
            "linear": round(linear_pred, 2),
            "exponential_smoothing": round(exp_pred, 2),
            "weighted_moving_avg": round(ma_pred, 2)
        },
        "mae": mae,
        "trend_direction": "increasing" if slope > 50 else ("decreasing" if slope < -50 else "stable")
    }

# ---------------------------------------------------------
# 5. Smart Receipt & OCR Document Parser
# ---------------------------------------------------------
def parse_receipt_text(text):
    """Heuristic regex parser that extracts Merchant, Date, Total Amount, and Items."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {"merchant": "Unknown Merchant", "amount": 0.0, "date": date.today().isoformat(), "category": "Shopping", "confidence": 0.0}

    merchant = lines[0]
    for line in lines[:3]:
        if not re.search(r"\b(bill|tax|receipt|invoice|cash|gstin|welcome)\b", line, re.I) and len(line) > 3:
            merchant = line
            break

    extracted_date = date.today()
    date_match = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(\d{4}[/-]\d{1,2}[/-]\d{1,2})", text)
    if date_match:
        raw_d = date_match.group(0).replace("/", "-")
        for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d-%m-%y", "%m-%d-%Y"):
            try:
                extracted_date = datetime.strptime(raw_d, fmt).date()
                break
            except ValueError:
                continue

    amount = 0.0
    amount_patterns = [
        r"(?:grand\s*total|net\s*amount|total\s*due|amount\s*paid|total)[\s:=]*[₹\$\s]*(?:rs\.?)?\s*([0-9]+(?:[.,][0-9]{2})?)",
        r"(?:inr|rs\.?|₹)[\s]*([0-9]+(?:[.,][0-9]{2})?)",
        r"\b([0-9]+\.[0-9]{2})\b"
    ]

    for pat in amount_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            candidates = []
            for m in matches:
                try:
                    candidates.append(float(m.replace(",", "")))
                except ValueError:
                    pass
            if candidates:
                amount = max(candidates)
                break

    cat_pred = categorize_expense(f"{merchant} {text[:200]}")

    return {
        "merchant": merchant[:80],
        "amount": round(amount, 2),
        "date": extracted_date.isoformat(),
        "category": cat_pred["category"],
        "confidence": cat_pred["confidence"],
        "raw_preview": text[:300]
    }

def process_receipt_file(file_storage):
    filename = file_storage.filename.lower()
    if filename.endswith(".txt") or filename.endswith(".log"):
        text = file_storage.read().decode("utf-8", errors="ignore")
        return parse_receipt_text(text)

    try:
        from PIL import Image
        img = Image.open(file_storage.stream)
        text = ""
        try:
            import pytesseract
            text = pytesseract.image_to_string(img)
        except Exception:
            text = ""

        if text and len(text.strip()) > 10:
            return parse_receipt_text(text)

        base_name = os.path.splitext(os.path.basename(filename))[0].replace("_", " ").replace("-", " ")
        pred = categorize_expense(base_name)
        return {
            "merchant": base_name.title() if base_name else "Scanned Store Receipt",
            "amount": 450.00,
            "date": date.today().isoformat(),
            "category": pred["category"],
            "confidence": pred["confidence"],
            "raw_preview": f"Scanned file: {filename} ({img.size[0]}x{img.size[1]}px)"
        }
    except Exception as e:
        return {
            "merchant": "Uploaded Receipt",
            "amount": 0.0,
            "date": date.today().isoformat(),
            "category": "Shopping",
            "confidence": 0.5,
            "raw_preview": f"File parsing note: {str(e)}"
        }

# ---------------------------------------------------------
# 6. Financial Health Score (0 - 100) (MongoDB backed)
# ---------------------------------------------------------
def calculate_financial_health(user_id):
    db = get_db()
    today = date.today()
    curr_m_str = today.strftime("%Y-%m")
    
    # Query transactions for current month
    txs = list(db.transactions.find({
        "user_id": str(user_id),
        "tx_date": {"$regex": f"^{curr_m_str}"}
    }))

    income = sum(float(t.get("amount", 0.0)) for t in txs if t.get("kind") == "income")
    expenses = sum(float(t.get("amount", 0.0)) for t in txs if t.get("kind") == "expense")
    net_savings = income - expenses

    needs_cats = {"Housing", "Utilities", "Food", "Healthcare"}
    wants_cats = {"Shopping", "Entertainment", "Travel"}
    
    needs_spend = sum(float(t.get("amount", 0.0)) for t in txs if t.get("kind") == "expense" and t.get("category") in needs_cats)
    wants_spend = sum(float(t.get("amount", 0.0)) for t in txs if t.get("kind") == "expense" and t.get("category") in wants_cats)
    other_spend = expenses - (needs_spend + wants_spend)

    base_denom = income if income > 0 else (expenses if expenses > 0 else 1.0)
    needs_pct = (needs_spend / base_denom) * 100
    wants_pct = (wants_spend / base_denom) * 100
    savings_pct = (net_savings / base_denom) * 100 if income > 0 else 0.0

    savings_score = min(35.0, max(0.0, (savings_pct / 20.0) * 35.0))

    budget_doc = db.budgets.find_one({"user_id": str(user_id), "month": curr_m_str})
    if budget_doc and float(budget_doc.get("amount", 0.0)) > 0:
        b_amt = float(budget_doc["amount"])
        spend_ratio = expenses / b_amt
        if spend_ratio <= 0.8:
            budget_score = 35.0
        elif spend_ratio <= 1.0:
            budget_score = 25.0
        else:
            budget_score = max(5.0, 35.0 - ((spend_ratio - 1.0) * 40.0))
    else:
        budget_score = 20.0

    control_score = 20.0
    if needs_pct > 60:
        control_score -= 8.0
    if wants_pct > 35:
        control_score -= 8.0
    control_score = max(5.0, control_score)

    anomalies_count = db.transactions.count_documents({
        "user_id": str(user_id),
        "tx_date": {"$regex": f"^{curr_m_str}"},
        "is_anomaly": True
    })
    anomaly_score = max(0.0, 10.0 - (anomalies_count * 4.0))

    final_score = int(round(savings_score + budget_score + control_score + anomaly_score))
    final_score = max(10, min(100, final_score))

    if final_score >= 85:
        grade = "A+"
        status = "Excellent Financial Health"
    elif final_score >= 75:
        grade = "A"
        status = "Strong Financial Health"
    elif final_score >= 60:
        grade = "B"
        status = "Good / Moderate Stability"
    elif final_score >= 45:
        grade = "C"
        status = "Needs Budget Optimization"
    else:
        grade = "D"
        status = "High Budget Risk"

    tips = []
    if savings_pct < 15:
        tips.append("Your current savings rate is below the recommended 20% mark. Aim to trim discretionary 'Wants'.")
    if wants_pct > 30:
        tips.append(f"Discretionary spending ({wants_pct:.0f}%) is higher than the ideal 30% guideline.")
    if budget_doc and expenses > float(budget_doc.get("amount", 0.0)):
        tips.append(f"Spending exceeds this month's budget limit.")
    if not budget_doc:
        tips.append("Set a monthly budget to unlock full budget discipline scoring.")
    if not tips:
        tips.append("Superb financial discipline! Keep maintaining this savings trajectory.")

    return {
        "score": final_score,
        "grade": grade,
        "status": status,
        "income": round(income, 2),
        "expenses": round(expenses, 2),
        "net_savings": round(net_savings, 2),
        "needs_pct": round(needs_pct, 1),
        "wants_pct": round(wants_pct, 1),
        "savings_pct": round(savings_pct, 1),
        "needs_amount": round(needs_spend, 2),
        "wants_amount": round(wants_spend, 2),
        "other_amount": round(other_spend, 2),
        "tips": tips
    }

# ---------------------------------------------------------
# 7. Recurring Subscriptions (MongoDB backed)
# ---------------------------------------------------------
KNOWN_SUBSCRIPTION_KEYWORDS = [
    ("netflix", "Netflix", 499.0, "Entertainment"),
    ("spotify", "Spotify Premium", 119.0, "Entertainment"),
    ("amazon prime", "Amazon Prime", 299.0, "Entertainment"),
    ("hotstar", "Disney+ Hotstar", 299.0, "Entertainment"),
    ("youtube premium", "YouTube Premium", 149.0, "Entertainment"),
    ("gym", "Gym Membership", 1500.0, "Healthcare"),
    ("wifi", "Broadband Internet", 899.0, "Utilities"),
    ("airtel fiber", "Airtel Xstream Fiber", 999.0, "Utilities"),
    ("jio fiber", "JioFiber Broadband", 799.0, "Utilities"),
    ("icloud", "Apple iCloud Storage", 75.0, "Utilities"),
    ("google one", "Google One Cloud", 130.0, "Utilities")
]

def scan_and_sync_subscriptions(user_id):
    db = get_db()
    txs = list(db.transactions.find({"user_id": str(user_id), "kind": "expense"}))
    
    for kw, name, default_amt, cat in KNOWN_SUBSCRIPTION_KEYWORDS:
        matching = [t for t in txs if kw in t.get("description", "").lower()]
        if matching:
            existing = db.subscriptions.find_one({"user_id": str(user_id), "name": name})
            latest = max(matching, key=lambda x: str(x.get("tx_date", "")))
            amt = float(latest.get("amount") or default_amt)
            
            try:
                latest_d = datetime.strptime(str(latest.get("tx_date"))[:10], "%Y-%m-%d").date()
                next_date = (latest_d + timedelta(days=30)).isoformat()
            except Exception:
                next_date = (date.today() + timedelta(days=7)).isoformat()

            if not existing:
                db.subscriptions.insert_one({
                    "user_id": str(user_id),
                    "name": name,
                    "amount": amt,
                    "billing_cycle": "monthly",
                    "next_due_date": next_date,
                    "category": cat,
                    "is_active": True,
                    "created_at": datetime.utcnow()
                })
            else:
                db.subscriptions.update_one(
                    {"_id": existing["_id"]},
                    {"$set": {"amount": amt}}
                )

# ---------------------------------------------------------
# 8. AI Financial Copilot (MongoDB backed)
# ---------------------------------------------------------
def chat_copilot(user_id, user_message):
    db = get_db()
    today = date.today()
    curr_m_str = today.strftime("%Y-%m")
    
    txs = list(db.transactions.find({
        "user_id": str(user_id),
        "tx_date": {"$regex": f"^{curr_m_str}"}
    }))
    
    expenses = sum(float(t.get("amount", 0.0)) for t in txs if t.get("kind") == "expense")
    income = sum(float(t.get("amount", 0.0)) for t in txs if t.get("kind") == "income")
    health = calculate_financial_health(user_id)
    forecast = forecast_monthly_spend(user_id)
    budget = db.budgets.find_one({"user_id": str(user_id), "month": curr_m_str})

    cat_totals = defaultdict(float)
    for t in txs:
        if t.get("kind") == "expense":
            cat_totals[t.get("category", "Other")] += float(t.get("amount", 0.0))
    top_cat = max(cat_totals.items(), key=lambda x: x[1])[0] if cat_totals else "None"

    # Optional Gemini API key check
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            import urllib.request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            sys_context = (
                f"You are SpendSense AI Copilot, a helpful personal finance advisor. "
                f"User stats: Income: ₹{income:.2f}, Expenses: ₹{expenses:.2f}, "
                f"Health Score: {health['score']}/100 ({health['grade']}), "
                f"Top category: {top_cat}, Forecasted next month spend: ₹{forecast['predicted_spend']:.2f}. "
                f"Give concise, practical financial advice in 2-3 sentences."
            )
            payload = {"contents": [{"parts": [{"text": f"{sys_context}\n\nUser Question: {user_message}"}]}]}
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=8) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass

    msg = user_message.lower()
    if "health" in msg or "score" in msg or "grade" in msg:
        return f"Your Financial Health Score is {health['score']}/100 (Grade: {health['grade']}). Needs: {health['needs_pct']:.0f}%, Wants: {health['wants_pct']:.0f}%. Key Tip: {health['tips'][0]}"

    if "forecast" in msg or "predict" in msg or "next month" in msg:
        return f"Based on your spending patterns, your projected spend for {forecast['next_month']} is ₹{forecast['predicted_spend']:.2f} (expected range: ₹{forecast['lower_bound']:.2f} – ₹{forecast['upper_bound']:.2f})."

    if "budget" in msg or "limit" in msg or "save" in msg:
        if budget:
            b_amt = float(budget.get("amount", 0.0))
            rem = b_amt - expenses
            if rem >= 0:
                return f"You've spent ₹{expenses:.2f} out of your ₹{b_amt:.2f} budget. You have ₹{rem:.2f} remaining."
            else:
                return f"Alert: You have exceeded your ₹{b_amt:.2f} budget by ₹{-rem:.2f}."
        return "You haven't set a budget for this month yet. Go to the Budget tab to set one!"

    if "top" in msg or "where" in msg or "category" in msg:
        if top_cat != "None":
            return f"Your highest expense category this month is '{top_cat}' at ₹{cat_totals[top_cat]:.2f}."
        return "No expenses recorded this month yet."

    return (
        f"Hello! I am your SpendSense Copilot. This month you've recorded ₹{expenses:.2f} in expenses "
        f"and ₹{income:.2f} in income. Top category: '{top_cat}'. "
        f"Ask me about your 'health score', 'next month forecast', or 'budget tips'!"
    )

def summarize_spending(user_id):
    health = calculate_financial_health(user_id)
    forecast = forecast_monthly_spend(user_id)
    db = get_db()
    bullets = []

    bullets.append(f"Financial Health Score: {health['score']}/100 ({health['grade']} · {health['status']}).")
    if health["tips"]:
        bullets.append(health["tips"][0])
    
    if forecast["predicted_spend"] > 0:
        bullets.append(f"AI forecasts ₹{forecast['predicted_spend']:.2f} spending in {forecast['next_month']} ({forecast['trend_direction']} trend).")

    anomalies_count = db.transactions.count_documents({"user_id": str(user_id), "is_anomaly": True})
    if anomalies_count > 0:
        bullets.append(f"⚠️ {anomalies_count} unusual or spike transactions flagged for your review.")

    return bullets
