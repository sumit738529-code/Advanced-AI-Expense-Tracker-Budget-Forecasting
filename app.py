import os
from datetime import date, datetime, timedelta
from functools import wraps
from io import BytesIO
from collections import defaultdict
import csv
import json

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, send_file, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId

from db_mongo import (
    get_db, is_mock_db,
    UserDoc, TransactionDoc, BudgetDoc, GoalDoc, SubscriptionDoc
)
from ai_engine import (
    categorize_expense, forecast_monthly_spend, summarize_spending,
    calculate_financial_health, detect_transaction_anomaly,
    parse_receipt_text, process_receipt_file,
    scan_and_sync_subscriptions, chat_copilot,
    CATEGORIES
)
from pdf_generator import generate_pdf_statement

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "spendsense-mongodb-secret-key-2026")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# Initialize MongoDB connection on startup
with app.app_context():
    get_db()


def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please sign in to access your financial dashboard.", "warning")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapped


def current_user():
    if "user_id" not in session:
        return None
    db = get_db()
    try:
        user_doc = db.users.find_one({"_id": ObjectId(session["user_id"])})
        return UserDoc(user_doc) if user_doc else None
    except Exception:
        return None


@app.context_processor
def inject_global_vars():
    user = current_user()
    return {
        "CATEGORIES": CATEGORIES,
        "current_user_obj": user,
        "active_currency": user.currency if user else "₹",
        "is_mock_db": is_mock_db()
    }


# ---------------------------------------------------------
# Authentication & Landing Routes
# ---------------------------------------------------------
@app.route("/")
def index():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return render_template("landing.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        db = get_db()
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        currency = request.form.get("currency", "₹")

        if not name or not email or len(password) < 8:
            flash("Please provide your name, valid email, and a password of at least 8 characters.", "danger")
            return render_template("register.html")

        if db.users.find_one({"email": email}):
            flash("An account with that email already exists. Please sign in.", "warning")
            return render_template("register.html")

        result = db.users.insert_one({
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "currency": currency,
            "monthly_income": 0.0,
            "created_at": datetime.utcnow()
        })
        session.clear()
        session["user_id"] = str(result.inserted_id)
        flash(f"Account created successfully! Welcome to SpendSense, {name.split()[0]}.", "success")
        return redirect(url_for("dashboard"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        db = get_db()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user_doc = db.users.find_one({"email": email})

        if not user_doc or not check_password_hash(user_doc["password_hash"], password):
            flash("Incorrect email or password. Please verify and try again.", "danger")
            return render_template("login.html")

        session.clear()
        session["user_id"] = str(user_doc["_id"])
        flash(f"Welcome back, {user_doc.get('name', 'User').split()[0]}!", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("index"))


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    user = current_user()
    db = get_db()
    if request.method == "POST":
        name = request.form.get("name", user.name).strip()
        currency = request.form.get("currency", user.currency)
        try:
            monthly_income = float(request.form.get("monthly_income", user.monthly_income))
        except ValueError:
            monthly_income = user.monthly_income

        update_fields = {
            "name": name,
            "currency": currency,
            "monthly_income": monthly_income
        }

        new_password = request.form.get("new_password", "").strip()
        if new_password:
            if len(new_password) < 8:
                flash("New password must be at least 8 characters.", "danger")
                return render_template("settings.html", user=user)
            update_fields["password_hash"] = generate_password_hash(new_password)

        db.users.update_one({"_id": ObjectId(user.id)}, {"$set": update_fields})
        flash("Your profile preferences have been updated.", "success")
        return redirect(url_for("settings"))

    return render_template("settings.html", user=user)


# ---------------------------------------------------------
# Dashboard Route
# ---------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    db = get_db()
    today = date.today()
    curr_m_str = today.strftime("%Y-%m")

    # Current month transactions
    month_txs_raw = list(db.transactions.find({
        "user_id": str(user.id),
        "tx_date": {"$regex": f"^{curr_m_str}"}
    }))
    
    expenses = sum(float(t.get("amount", 0.0)) for t in month_txs_raw if t.get("kind") == "expense")
    income = sum(float(t.get("amount", 0.0)) for t in month_txs_raw if t.get("kind") == "income")
    net_savings = income - expenses

    # Monthly Budget
    budget_doc = db.budgets.find_one({"user_id": str(user.id), "month": curr_m_str})
    budget_amount = float(budget_doc.get("amount", 0.0)) if budget_doc else 0.0

    # Category breakdown
    cats = defaultdict(float)
    for t in month_txs_raw:
        if t.get("kind") == "expense":
            cats[t.get("category", "Other")] += float(t.get("amount", 0.0))

    # Recent transactions (last 7)
    recent_raw = list(db.transactions.find({"user_id": str(user.id)}).sort("tx_date", -1).limit(7))
    recent_txs = [TransactionDoc(t) for t in recent_raw]

    # Anomalies
    anom_raw = list(db.transactions.find({"user_id": str(user.id), "is_anomaly": True}).sort("tx_date", -1).limit(3))
    anomalies = [TransactionDoc(t) for t in anom_raw]

    # AI Forecast, Financial Health Score & Insights
    forecast = forecast_monthly_spend(user.id)
    health = calculate_financial_health(user.id)
    insights = summarize_spending(user.id)

    # Active Goals & Upcoming Subscriptions
    goals_raw = list(db.goals.find({"user_id": str(user.id), "status": "active"}).limit(3))
    goals = [GoalDoc(g) for g in goals_raw]

    subs_raw = list(db.subscriptions.find({"user_id": str(user.id), "is_active": True}).sort("next_due_date", 1).limit(3))
    upcoming_subs = [SubscriptionDoc(s) for s in subs_raw]

    return render_template(
        "dashboard.html",
        user=user,
        today=today,
        expenses=expenses,
        income=income,
        net_savings=net_savings,
        budget_amount=budget_amount,
        categories=dict(cats),
        transactions=recent_txs,
        anomalies=anomalies,
        forecast=forecast,
        health=health,
        insights=insights,
        goals=goals,
        upcoming_subs=upcoming_subs
    )


# ---------------------------------------------------------
# Transactions Management Routes
# ---------------------------------------------------------
@app.route("/transactions")
@login_required
def transactions():
    user = current_user()
    db = get_db()
    
    query = {"user_id": str(user.id)}
    category = request.args.get("category", "")
    kind = request.args.get("kind", "")
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    anomalies_only = request.args.get("anomalies_only", "")

    if category:
        query["category"] = category
    if kind in ("expense", "income"):
        query["kind"] = kind
    if anomalies_only == "1":
        query["is_anomaly"] = True

    date_filters = {}
    if start:
        date_filters["$gte"] = start
    if end:
        date_filters["$lte"] = end
    if date_filters:
        query["tx_date"] = date_filters

    raw_txs = list(db.transactions.find(query).sort("tx_date", -1))
    tx_list = [TransactionDoc(t) for t in raw_txs]

    total_filtered = sum(t.amount for t in tx_list if t.kind == "expense")
    total_income_filtered = sum(t.amount for t in tx_list if t.kind == "income")

    return render_template(
        "transactions.html",
        transactions=tx_list,
        categories=CATEGORIES,
        total_filtered=total_filtered,
        total_income_filtered=total_income_filtered
    )


@app.route("/transactions/new", methods=["GET", "POST"])
@login_required
def new_transaction():
    user = current_user()
    db = get_db()

    if request.method == "POST":
        description = request.form.get("description", "").strip()
        kind = request.form.get("kind", "expense")
        payment_method = request.form.get("payment_method", "UPI / Card")
        notes = request.form.get("notes", "").strip()
        tx_date_str = request.form.get("tx_date", "").strip()

        try:
            amount = round(float(request.form.get("amount", "0")), 2)
            tx_date = datetime.strptime(tx_date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            flash("Please enter a valid monetary amount and date.", "danger")
            return redirect(url_for("new_transaction"))

        if amount <= 0 or not description or kind not in ("expense", "income"):
            flash("Amount must be positive and description cannot be empty.", "danger")
            return redirect(url_for("new_transaction"))

        predicted = categorize_expense(description)
        user_cat = request.form.get("category", "").strip()
        final_category = user_cat if user_cat in CATEGORIES else predicted["category"]

        is_anomaly = False
        anomaly_reason = None
        if kind == "expense":
            is_anomaly, anomaly_reason = detect_transaction_anomaly(
                user.id, amount, final_category, tx_date, description
            )

        db.transactions.insert_one({
            "user_id": str(user.id),
            "description": description,
            "amount": amount,
            "tx_date": tx_date.isoformat(),
            "kind": kind,
            "category": final_category,
            "ai_category": predicted["category"],
            "ai_confidence": predicted["confidence"],
            "is_anomaly": is_anomaly,
            "anomaly_reason": anomaly_reason,
            "payment_method": payment_method,
            "notes": notes,
            "created_at": datetime.utcnow()
        })

        if is_anomaly:
            flash(f"Transaction logged. ⚠️ AI flagged an anomaly: {anomaly_reason}", "warning")
        else:
            flash(f"Transaction saved! AI categorized as '{predicted['category']}' ({predicted['confidence']:.0%} confidence).", "success")

        return redirect(url_for("transactions"))

    return render_template(
        "transaction_form.html",
        categories=CATEGORIES,
        today=date.today().isoformat(),
        action="Add"
    )


@app.route("/transactions/<tx_id>/edit", methods=["GET", "POST"])
@login_required
def edit_transaction(tx_id):
    user = current_user()
    db = get_db()
    
    try:
        raw_doc = db.transactions.find_one({"_id": ObjectId(tx_id), "user_id": str(user.id)})
    except Exception:
        raw_doc = None
    if not raw_doc:
        flash("Transaction not found.", "danger")
        return redirect(url_for("transactions"))

    tx = TransactionDoc(raw_doc)

    if request.method == "POST":
        description = request.form.get("description", "").strip()
        kind = request.form.get("kind", "expense")
        payment_method = request.form.get("payment_method", "UPI / Card")
        category = request.form.get("category", tx.category)
        notes = request.form.get("notes", "").strip()
        tx_date_str = request.form.get("tx_date", "").strip()

        try:
            amount = round(float(request.form.get("amount", "0")), 2)
            tx_date = datetime.strptime(tx_date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            flash("Enter a valid amount and date.", "danger")
            return redirect(url_for("edit_transaction", tx_id=tx_id))

        if amount <= 0 or not description:
            flash("Amount must be positive and description cannot be empty.", "danger")
            return redirect(url_for("edit_transaction", tx_id=tx_id))

        final_cat = category if category in CATEGORIES else "Other"
        is_anom, reason = False, None
        if kind == "expense":
            is_anom, reason = detect_transaction_anomaly(user.id, amount, final_cat, tx_date, description)

        db.transactions.update_one(
            {"_id": ObjectId(tx_id), "user_id": str(user.id)},
            {"$set": {
                "description": description,
                "amount": amount,
                "tx_date": tx_date.isoformat(),
                "kind": kind,
                "category": final_cat,
                "payment_method": payment_method,
                "notes": notes,
                "is_anomaly": is_anom,
                "anomaly_reason": reason
            }}
        )

        flash("Transaction updated successfully.", "success")
        return redirect(url_for("transactions"))

    return render_template(
        "transaction_form.html",
        transaction=tx,
        categories=CATEGORIES,
        today=tx.tx_date.isoformat(),
        action="Edit"
    )


@app.route("/transactions/<tx_id>/delete", methods=["POST"])
@login_required
def delete_transaction(tx_id):
    user = current_user()
    db = get_db()
    try:
        db.transactions.delete_one({"_id": ObjectId(tx_id), "user_id": str(user.id)})
        flash("Transaction deleted successfully.", "success")
    except Exception as e:
        flash(f"Could not delete transaction: {e}", "danger")
    return redirect(url_for("transactions"))


# ---------------------------------------------------------
# Smart Receipt & OCR Scanner Routes
# ---------------------------------------------------------
@app.route("/receipt-scanner", methods=["GET", "POST"])
@login_required
def receipt_scanner():
    user = current_user()
    extracted_data = None

    if request.method == "POST":
        receipt_file = request.files.get("receipt_file")
        pasted_text = request.form.get("pasted_text", "").strip()

        if receipt_file and receipt_file.filename:
            extracted_data = process_receipt_file(receipt_file)
        elif pasted_text:
            extracted_data = parse_receipt_text(pasted_text)
        else:
            flash("Please either upload a receipt file/photo or paste receipt text.", "warning")

    return render_template(
        "receipt_scanner.html",
        extracted=extracted_data,
        categories=CATEGORIES,
        today=date.today().isoformat()
    )


# ---------------------------------------------------------
# Budgets & Category Limits Route
# ---------------------------------------------------------
@app.route("/budget", methods=["GET", "POST"])
@login_required
def budget():
    user = current_user()
    db = get_db()
    month = request.args.get("month", date.today().strftime("%Y-%m"))

    if request.method == "POST":
        month = request.form.get("month", date.today().strftime("%Y-%m"))
        try:
            amount = float(request.form.get("amount", "0"))
            datetime.strptime(month, "%Y-%m")
            if amount <= 0:
                raise ValueError()
        except ValueError:
            flash("Please provide a valid month and a budget greater than zero.", "danger")
            return redirect(url_for("budget", month=month))

        cat_limits = {}
        for c in CATEGORIES:
            limit_val = request.form.get(f"limit_{c}", "").strip()
            if limit_val:
                try:
                    c_amt = float(limit_val)
                    if c_amt > 0:
                        cat_limits[c] = c_amt
                except ValueError:
                    pass

        db.budgets.update_one(
            {"user_id": str(user.id), "month": month},
            {"$set": {
                "user_id": str(user.id),
                "month": month,
                "amount": amount,
                "category_limits": cat_limits,
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )

        flash("Monthly budget and category limits saved.", "success")
        return redirect(url_for("budget", month=month))

    raw_budget = db.budgets.find_one({"user_id": str(user.id), "month": month})
    budget_obj = BudgetDoc(raw_budget) if raw_budget else None

    # Calculate total spent and category-wise spent in this month
    cat_txs = list(db.transactions.find({
        "user_id": str(user.id),
        "kind": "expense",
        "tx_date": {"$regex": f"^{month}"}
    }))
    spent = sum(float(t.get("amount", 0.0)) for t in cat_txs)

    cat_spent = defaultdict(float)
    for t in cat_txs:
        cat_spent[t.get("category", "Other")] += float(t.get("amount", 0.0))

    category_limits = budget_obj.category_limits if budget_obj else {}

    return render_template(
        "budget.html",
        budget=budget_obj,
        month=month,
        spent=float(spent),
        category_limits=category_limits,
        cat_spent=dict(cat_spent)
    )


# ---------------------------------------------------------
# Savings Goals Route
# ---------------------------------------------------------
@app.route("/goals", methods=["GET", "POST"])
@login_required
def goals():
    user = current_user()
    db = get_db()

    if request.method == "POST":
        action = request.form.get("action")
        if action == "create":
            title = request.form.get("title", "").strip()
            try:
                target_amount = float(request.form.get("target_amount", "0"))
                current_amount = float(request.form.get("current_amount", "0"))
                target_date_str = request.form.get("target_date", "").strip()
                target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
                if target_amount <= 0:
                    raise ValueError()
            except ValueError:
                flash("Enter valid goal numbers and completion date.", "danger")
                return redirect(url_for("goals"))

            category = request.form.get("category", "General")
            db.goals.insert_one({
                "user_id": str(user.id),
                "title": title,
                "target_amount": target_amount,
                "current_amount": current_amount,
                "target_date": target_date.isoformat(),
                "category": category,
                "status": "active",
                "created_at": datetime.utcnow()
            })
            flash(f"Goal '{title}' created successfully!", "success")

        elif action == "contribute":
            goal_id = request.form.get("goal_id")
            try:
                contrib = float(request.form.get("contribution", "0"))
                g_raw = db.goals.find_one({"_id": ObjectId(goal_id), "user_id": str(user.id)})
                if g_raw:
                    new_curr = float(g_raw.get("current_amount", 0.0)) + contrib
                    status = "completed" if new_curr >= float(g_raw.get("target_amount", 0.0)) else g_raw.get("status", "active")
                    db.goals.update_one({"_id": ObjectId(goal_id)}, {"$set": {"current_amount": new_curr, "status": status}})
                    if status == "completed":
                        flash(f"🎉 Congratulations! You achieved your savings goal: '{g_raw.get('title')}'!", "success")
                    else:
                        flash(f"Added {user.currency}{contrib:.2f} towards '{g_raw.get('title')}'.", "success")
            except Exception as e:
                flash(f"Could not process contribution: {e}", "danger")

        elif action == "delete":
            goal_id = request.form.get("goal_id")
            try:
                db.goals.delete_one({"_id": ObjectId(goal_id), "user_id": str(user.id)})
                flash("Savings goal deleted.", "info")
            except Exception:
                pass

        return redirect(url_for("goals"))

    raw_goals = list(db.goals.find({"user_id": str(user.id)}).sort("target_date", 1))
    goals_list = [GoalDoc(g) for g in raw_goals]
    return render_template("goals.html", goals=goals_list)


# ---------------------------------------------------------
# Recurring Subscriptions Route
# ---------------------------------------------------------
@app.route("/subscriptions", methods=["GET", "POST"])
@login_required
def subscriptions():
    user = current_user()
    db = get_db()

    if request.method == "POST":
        action = request.form.get("action")
        if action == "sync":
            scan_and_sync_subscriptions(user.id)
            flash("Scanned your transaction records and synchronized detected subscriptions.", "success")
        elif action == "create":
            name = request.form.get("name", "").strip()
            try:
                amount = float(request.form.get("amount", "0"))
                next_due_str = request.form.get("next_due_date", "").strip()
                next_due = datetime.strptime(next_due_str, "%Y-%m-%d").date()
                if amount <= 0:
                    raise ValueError()
            except ValueError:
                flash("Enter a valid subscription amount and next due date.", "danger")
                return redirect(url_for("subscriptions"))

            billing_cycle = request.form.get("billing_cycle", "monthly")
            category = request.form.get("category", "Entertainment")
            db.subscriptions.insert_one({
                "user_id": str(user.id),
                "name": name,
                "amount": amount,
                "billing_cycle": billing_cycle,
                "next_due_date": next_due.isoformat(),
                "category": category,
                "is_active": True,
                "created_at": datetime.utcnow()
            })
            flash(f"Subscription '{name}' added.", "success")

        elif action == "delete":
            sub_id = request.form.get("sub_id")
            try:
                db.subscriptions.delete_one({"_id": ObjectId(sub_id), "user_id": str(user.id)})
                flash("Subscription removed.", "info")
            except Exception:
                pass

        return redirect(url_for("subscriptions"))

    raw_subs = list(db.subscriptions.find({"user_id": str(user.id)}).sort("next_due_date", 1))
    subs_list = [SubscriptionDoc(s) for s in raw_subs]

    monthly_burden = sum(s.amount if s.billing_cycle == "monthly" else (s.amount / 12.0) for s in subs_list if s.is_active)
    yearly_burden = monthly_burden * 12.0

    return render_template(
        "subscriptions.html",
        subscriptions=subs_list,
        monthly_burden=round(monthly_burden, 2),
        yearly_burden=round(yearly_burden, 2)
    )


# ---------------------------------------------------------
# Import & Export Routes
# ---------------------------------------------------------
@app.route("/import", methods=["GET", "POST"])
@login_required
def import_csv():
    user = current_user()
    db = get_db()

    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename.lower().endswith(".csv"):
            flash("Please upload a valid .csv file.", "danger")
            return redirect(url_for("import_csv"))
        try:
            content = file.stream.read().decode("utf-8-sig").splitlines()
            reader = csv.DictReader(content)
            required = {"date", "description", "amount"}
            if not reader.fieldnames or not required.issubset({x.strip().lower() for x in reader.fieldnames}):
                raise ValueError("CSV must include 'date', 'description', and 'amount' columns.")

            count = 0
            for row in reader:
                normalized = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
                try:
                    tx_date = datetime.strptime(normalized["date"], "%Y-%m-%d").date()
                except ValueError:
                    continue

                try:
                    amount = abs(float(normalized["amount"]))
                except ValueError:
                    continue

                if amount <= 0 or not normalized["description"]:
                    continue

                kind = normalized.get("type", "expense").lower()
                if kind not in ("expense", "income"):
                    kind = "expense"

                predicted = categorize_expense(normalized["description"])
                category = normalized.get("category") or predicted["category"]
                if category not in CATEGORIES:
                    category = predicted["category"]

                is_anom, reason = detect_transaction_anomaly(
                    user.id, amount, category, tx_date, normalized["description"]
                )

                db.transactions.insert_one({
                    "user_id": str(user.id),
                    "description": normalized["description"],
                    "amount": amount,
                    "tx_date": tx_date.isoformat(),
                    "kind": kind,
                    "category": category,
                    "ai_category": predicted["category"],
                    "ai_confidence": predicted["confidence"],
                    "is_anomaly": is_anom,
                    "anomaly_reason": reason,
                    "payment_method": "Imported",
                    "created_at": datetime.utcnow()
                })
                count += 1

            flash(f"Successfully imported and AI-categorized {count} transactions into MongoDB.", "success")
            return redirect(url_for("transactions"))
        except Exception as exc:
            flash(f"Import failed: {exc}", "danger")

    return render_template("import.html")


@app.route("/export/csv")
@login_required
def export_csv():
    user = current_user()
    db = get_db()
    raw_txs = list(db.transactions.find({"user_id": str(user.id)}).sort("tx_date", -1))
    
    import io
    text_stream = io.StringIO()
    writer = csv.writer(text_stream)
    writer.writerow([
        "date", "description", "amount", "type", "category",
        "ai_category", "ai_confidence", "is_anomaly", "anomaly_reason", "payment_method"
    ])
    for t in raw_txs:
        writer.writerow([
            t.get("tx_date"),
            t.get("description"),
            f"{float(t.get('amount', 0.0)):.2f}",
            t.get("kind"),
            t.get("category"),
            t.get("ai_category"),
            f"{float(t.get('ai_confidence', 0.0)):.2f}",
            "Yes" if t.get("is_anomaly") else "No",
            t.get("anomaly_reason") or "",
            t.get("payment_method", "UPI / Card")
        ])
    return send_file(
        BytesIO(text_stream.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"spendsense_statement_{date.today().strftime('%Y%m%d')}.csv"
    )


@app.route("/export/pdf")
@login_required
def export_pdf():
    user = current_user()
    db = get_db()
    raw_txs = list(db.transactions.find({"user_id": str(user.id)}).sort("tx_date", -1))
    tx_docs = [TransactionDoc(t) for t in raw_txs]
    health = calculate_financial_health(user.id)
    forecast = forecast_monthly_spend(user.id)
    pdf_buffer = generate_pdf_statement(user, tx_docs, health, forecast)
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"spendsense_statement_{date.today().strftime('%Y%m%d')}.pdf"
    )


# ---------------------------------------------------------
# REST APIs for Dynamic UI & AI Copilot
# ---------------------------------------------------------
@app.route("/api/categorize", methods=["POST"])
@login_required
def api_categorize():
    data = request.get_json(silent=True) or {}
    text = data.get("description", "")
    result = categorize_expense(text)
    return jsonify(result)


@app.route("/api/forecast")
@login_required
def api_forecast():
    user = current_user()
    return jsonify({
        "forecast": forecast_monthly_spend(user.id),
        "insights": summarize_spending(user.id),
        "health": calculate_financial_health(user.id)
    })


@app.route("/api/copilot", methods=["POST"])
@login_required
def api_copilot():
    user = current_user()
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"reply": "Please ask me a financial question."})
    reply = chat_copilot(user.id, message)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
