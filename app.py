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
from werkzeug.utils import secure_filename
from sqlalchemy import func, extract

from models import db, User, Transaction, Budget, SavingsGoal, Subscription
from ai_engine import (
    categorize_expense, forecast_monthly_spend, summarize_spending,
    calculate_financial_health, detect_transaction_anomaly,
    audit_user_anomalies, parse_receipt_text, process_receipt_file,
    get_model_evaluation, scan_and_sync_subscriptions, chat_copilot,
    CATEGORIES
)
from pdf_generator import generate_pdf_statement
from seed_data import seed_demo_data_for_user, ensure_demo_user

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "spendsense-dev-secret-key-final-year-2026")

# Database URI configuration with Render PostgreSQL compatibility
raw_db_url = os.environ.get("DATABASE_URL", "sqlite:///expense_ai.db")
if raw_db_url.startswith("postgres://"):
    raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = raw_db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB max upload size

db.init_app(app)

with app.app_context():
    db.create_all()
    # Create default demo user for seamless examiner/evaluator logins
    try:
        ensure_demo_user()
    except Exception as e:
        print(f"Demo user init notice: {e}")


def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please sign in to access your financial dashboard.", "warning")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapped


def current_user():
    return db.session.get(User, session.get("user_id"))


@app.context_processor
def inject_global_vars():
    user = current_user() if session.get("user_id") else None
    return {
        "CATEGORIES": CATEGORIES,
        "current_user_obj": user,
        "active_currency": user.currency if user else "₹"
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
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        currency = request.form.get("currency", "₹")
        
        if not name or not email or len(password) < 8:
            flash("Please provide your name, valid email, and a password of at least 8 characters.", "danger")
            return render_template("register.html")

        if User.query.filter_by(email=email).first():
            flash("An account with that email already exists. Please sign in.", "warning")
            return render_template("register.html")

        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            currency=currency
        )
        db.session.add(user)
        db.session.commit()
        session["user_id"] = user.id
        flash(f"Account created successfully! Welcome to SpendSense AI, {name.split()[0]}.", "success")
        return redirect(url_for("dashboard"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password_hash, password):
            flash("Incorrect email or password. Please verify and try again.", "danger")
            return render_template("login.html")

        session.clear()
        session["user_id"] = user.id
        flash(f"Welcome back, {user.name.split()[0]}!", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/demo-login")
def demo_login():
    """1-Click login for viva examiners and professors."""
    user = ensure_demo_user()
    session.clear()
    session["user_id"] = user.id
    flash("Logged in with Viva Demo Account (pre-loaded with 6 months of AI analyzed transactions).", "info")
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("index"))


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    user = current_user()
    if request.method == "POST":
        user.name = request.form.get("name", user.name).strip()
        user.currency = request.form.get("currency", user.currency)
        try:
            user.monthly_income = float(request.form.get("monthly_income", user.monthly_income))
        except ValueError:
            pass

        new_password = request.form.get("new_password", "").strip()
        if new_password:
            if len(new_password) < 8:
                flash("New password must be at least 8 characters.", "danger")
                return render_template("settings.html", user=user)
            user.password_hash = generate_password_hash(new_password)

        db.session.commit()
        flash("Your profile preferences have been updated.", "success")
        return redirect(url_for("settings"))

    return render_template("settings.html", user=user)


# ---------------------------------------------------------
# Core Dashboard Route
# ---------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    today = date.today()
    month_start = today.replace(day=1)

    # Transactions for current month
    month_txs = Transaction.query.filter(
        Transaction.user_id == user.id,
        Transaction.tx_date >= month_start,
        Transaction.tx_date <= today
    ).all()

    expenses = sum(t.amount for t in month_txs if t.kind == "expense")
    income = sum(t.amount for t in month_txs if t.kind == "income")
    net_savings = income - expenses

    # Monthly Budget
    budget_record = Budget.query.filter_by(user_id=user.id, month=today.strftime("%Y-%m")).first()
    budget_amount = budget_record.amount if budget_record else 0.0

    # Category breakdown for current month
    cats = defaultdict(float)
    for t in month_txs:
        if t.kind == "expense":
            cats[t.category] += t.amount

    # Recent transactions
    recent_txs = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.tx_date.desc()).limit(7).all()

    # AI Anomaly check
    anomalies = Transaction.query.filter_by(user_id=user.id, is_anomaly=True).order_by(Transaction.tx_date.desc()).limit(3).all()

    # AI Forecast, Financial Health Score & Insights
    forecast = forecast_monthly_spend(user.id)
    health = calculate_financial_health(user.id)
    insights = summarize_spending(user.id)

    # Active savings goals & Subscriptions
    goals = SavingsGoal.query.filter_by(user_id=user.id, status="active").limit(3).all()
    upcoming_subs = Subscription.query.filter_by(user_id=user.id, is_active=True).order_by(Subscription.next_due_date.asc()).limit(3).all()

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
    query = Transaction.query.filter_by(user_id=user.id)

    category = request.args.get("category", "")
    kind = request.args.get("kind", "")
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    anomalies_only = request.args.get("anomalies_only", "")

    if category:
        query = query.filter_by(category=category)
    if kind in ("expense", "income"):
        query = query.filter_by(kind=kind)
    if anomalies_only == "1":
        query = query.filter_by(is_anomaly=True)

    try:
        if start:
            query = query.filter(Transaction.tx_date >= datetime.strptime(start, "%Y-%m-%d").date())
        if end:
            query = query.filter(Transaction.tx_date <= datetime.strptime(end, "%Y-%m-%d").date())
    except ValueError:
        flash("Invalid date filter criteria applied.", "warning")

    rows = query.order_by(Transaction.tx_date.desc()).all()
    total_filtered = sum(t.amount for t in rows if t.kind == "expense")
    total_income_filtered = sum(t.amount for t in rows if t.kind == "income")

    return render_template(
        "transactions.html",
        transactions=rows,
        categories=CATEGORIES,
        total_filtered=total_filtered,
        total_income_filtered=total_income_filtered
    )


@app.route("/transactions/new", methods=["GET", "POST"])
@login_required
def new_transaction():
    user = current_user()
    if request.method == "POST":
        description = request.form.get("description", "").strip()
        kind = request.form.get("kind", "expense")
        payment_method = request.form.get("payment_method", "UPI / Card")
        notes = request.form.get("notes", "").strip()

        try:
            amount = round(float(request.form.get("amount", "0")), 2)
            tx_date = datetime.strptime(request.form.get("tx_date", ""), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            flash("Please enter a valid monetary amount and date.", "danger")
            return redirect(url_for("new_transaction"))

        if amount <= 0 or not description or kind not in ("expense", "income"):
            flash("Amount must be positive and description cannot be empty.", "danger")
            return redirect(url_for("new_transaction"))

        # AI Categorization
        predicted = categorize_expense(description)
        user_cat = request.form.get("category", "").strip()
        final_category = user_cat if user_cat in CATEGORIES else predicted["category"]

        # AI Anomaly check for expenses
        is_anomaly = False
        anomaly_reason = None
        if kind == "expense":
            is_anomaly, anomaly_reason = detect_transaction_anomaly(
                user.id, amount, final_category, tx_date, description
            )

        tx = Transaction(
            user_id=user.id,
            description=description,
            amount=amount,
            tx_date=tx_date,
            kind=kind,
            category=final_category,
            ai_category=predicted["category"],
            ai_confidence=predicted["confidence"],
            is_anomaly=is_anomaly,
            anomaly_reason=anomaly_reason,
            payment_method=payment_method,
            notes=notes
        )
        db.session.add(tx)
        db.session.commit()

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


@app.route("/transactions/<int:tx_id>/edit", methods=["GET", "POST"])
@login_required
def edit_transaction(tx_id):
    user = current_user()
    tx = Transaction.query.filter_by(id=tx_id, user_id=user.id).first_or_404()

    if request.method == "POST":
        description = request.form.get("description", "").strip()
        kind = request.form.get("kind", "expense")
        payment_method = request.form.get("payment_method", "UPI / Card")
        category = request.form.get("category", tx.category)
        notes = request.form.get("notes", "").strip()

        try:
            amount = round(float(request.form.get("amount", "0")), 2)
            tx_date = datetime.strptime(request.form.get("tx_date", ""), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            flash("Enter a valid amount and date.", "danger")
            return redirect(url_for("edit_transaction", tx_id=tx.id))

        if amount <= 0 or not description:
            flash("Amount must be positive and description cannot be empty.", "danger")
            return redirect(url_for("edit_transaction", tx_id=tx.id))

        tx.description = description
        tx.amount = amount
        tx.tx_date = tx_date
        tx.kind = kind
        tx.category = category if category in CATEGORIES else "Other"
        tx.payment_method = payment_method
        tx.notes = notes

        # Re-evaluate anomaly
        if kind == "expense":
            is_anom, reason = detect_transaction_anomaly(user.id, amount, tx.category, tx_date, description)
            tx.is_anomaly = is_anom
            tx.anomaly_reason = reason

        db.session.commit()
        flash("Transaction updated successfully.", "success")
        return redirect(url_for("transactions"))

    return render_template(
        "transaction_form.html",
        transaction=tx,
        categories=CATEGORIES,
        today=tx.tx_date.isoformat(),
        action="Edit"
    )


@app.route("/transactions/<int:tx_id>/delete", methods=["POST"])
@login_required
def delete_transaction(tx_id):
    tx = Transaction.query.filter_by(id=tx_id, user_id=session["user_id"]).first_or_404()
    db.session.delete(tx)
    db.session.commit()
    flash("Transaction deleted successfully.", "success")
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

        # Collect category-wise budget limits
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

        item = Budget.query.filter_by(user_id=user.id, month=month).first()
        if item:
            item.amount = amount
            item.category_limits = json.dumps(cat_limits)
        else:
            db.session.add(Budget(
                user_id=user.id,
                month=month,
                amount=amount,
                category_limits=json.dumps(cat_limits)
            ))

        db.session.commit()
        flash("Monthly budget and category limits saved.", "success")
        return redirect(url_for("budget", month=month))

    item = Budget.query.filter_by(user_id=user.id, month=month).first()
    try:
        first = datetime.strptime(month, "%Y-%m").date().replace(day=1)
    except ValueError:
        first = date.today().replace(day=1)
        month = first.strftime("%Y-%m")

    # Spent in month
    spent = db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0)).filter(
        Transaction.user_id == user.id,
        Transaction.kind == "expense",
        extract("year", Transaction.tx_date) == first.year,
        extract("month", Transaction.tx_date) == first.month
    ).scalar() or 0.0

    # Category breakdown for this month
    cat_txs = Transaction.query.filter(
        Transaction.user_id == user.id,
        Transaction.kind == "expense",
        extract("year", Transaction.tx_date) == first.year,
        extract("month", Transaction.tx_date) == first.month
    ).all()
    cat_spent = defaultdict(float)
    for t in cat_txs:
        cat_spent[t.category] += t.amount

    category_limits = json.loads(item.category_limits) if (item and item.category_limits) else {}

    return render_template(
        "budget.html",
        budget=item,
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
    if request.method == "POST":
        action = request.form.get("action")
        if action == "create":
            title = request.form.get("title", "").strip()
            try:
                target_amount = float(request.form.get("target_amount", "0"))
                current_amount = float(request.form.get("current_amount", "0"))
                target_date = datetime.strptime(request.form.get("target_date", ""), "%Y-%m-%d").date()
                if target_amount <= 0:
                    raise ValueError()
            except ValueError:
                flash("Enter valid goal numbers and completion date.", "danger")
                return redirect(url_for("goals"))

            category = request.form.get("category", "General")
            db.session.add(SavingsGoal(
                user_id=user.id,
                title=title,
                target_amount=target_amount,
                current_amount=current_amount,
                target_date=target_date,
                category=category
            ))
            db.session.commit()
            flash(f"Goal '{title}' created successfully!", "success")

        elif action == "contribute":
            goal_id = request.form.get("goal_id")
            try:
                contrib = float(request.form.get("contribution", "0"))
                g = SavingsGoal.query.filter_by(id=goal_id, user_id=user.id).first_or_404()
                g.current_amount += contrib
                if g.current_amount >= g.target_amount:
                    g.status = "completed"
                    flash(f"🎉 Congratulations! You achieved your savings goal: '{g.title}'!", "success")
                else:
                    flash(f"Added {user.currency}{contrib:.2f} towards '{g.title}'.", "success")
                db.session.commit()
            except ValueError:
                flash("Enter a valid contribution amount.", "danger")

        elif action == "delete":
            goal_id = request.form.get("goal_id")
            g = SavingsGoal.query.filter_by(id=goal_id, user_id=user.id).first_or_404()
            db.session.delete(g)
            db.session.commit()
            flash("Savings goal deleted.", "info")

        return redirect(url_for("goals"))

    all_goals = SavingsGoal.query.filter_by(user_id=user.id).order_by(SavingsGoal.status.asc(), SavingsGoal.target_date.asc()).all()
    return render_template("goals.html", goals=all_goals)


# ---------------------------------------------------------
# Recurring Subscriptions Route
# ---------------------------------------------------------
@app.route("/subscriptions", methods=["GET", "POST"])
@login_required
def subscriptions():
    user = current_user()
    if request.method == "POST":
        action = request.form.get("action")
        if action == "sync":
            # Auto scan user history to identify recurring patterns
            scan_and_sync_subscriptions(user.id)
            flash("Scanned your transaction records and synchronized detected subscriptions.", "success")
        elif action == "create":
            name = request.form.get("name", "").strip()
            try:
                amount = float(request.form.get("amount", "0"))
                next_due = datetime.strptime(request.form.get("next_due_date", ""), "%Y-%m-%d").date()
                if amount <= 0:
                    raise ValueError()
            except ValueError:
                flash("Enter a valid subscription amount and next due date.", "danger")
                return redirect(url_for("subscriptions"))

            billing_cycle = request.form.get("billing_cycle", "monthly")
            category = request.form.get("category", "Entertainment")
            db.session.add(Subscription(
                user_id=user.id,
                name=name,
                amount=amount,
                billing_cycle=billing_cycle,
                next_due_date=next_due,
                category=category
            ))
            db.session.commit()
            flash(f"Subscription '{name}' added.", "success")

        elif action == "delete":
            sub_id = request.form.get("sub_id")
            s = Subscription.query.filter_by(id=sub_id, user_id=user.id).first_or_404()
            db.session.delete(s)
            db.session.commit()
            flash("Subscription removed.", "info")

        return redirect(url_for("subscriptions"))

    all_subs = Subscription.query.filter_by(user_id=user.id).order_by(Subscription.next_due_date.asc()).all()
    monthly_burden = sum(s.amount if s.billing_cycle == "monthly" else (s.amount / 12.0) for s in all_subs if s.is_active)
    yearly_burden = monthly_burden * 12.0

    return render_template(
        "subscriptions.html",
        subscriptions=all_subs,
        monthly_burden=round(monthly_burden, 2),
        yearly_burden=round(yearly_burden, 2)
    )


# ---------------------------------------------------------
# AI Model & Viva Defense Lab Route
# ---------------------------------------------------------
@app.route("/viva-lab")
@login_required
def viva_lab():
    user = current_user()
    metrics = get_model_evaluation()
    forecast = forecast_monthly_spend(user.id)
    return render_template(
        "viva_lab.html",
        metrics=metrics,
        forecast=forecast
    )


# ---------------------------------------------------------
# Demo Data Seeder (1-Click Viva Showcase)
# ---------------------------------------------------------
@app.route("/demo-seed", methods=["POST"])
@login_required
def demo_seed():
    user = current_user()
    seed_demo_data_for_user(user.id)
    flash("🎉 Successfully seeded 6 months of realistic financial transactions, subscriptions, budgets, and anomalies! Ready for viva demonstration.", "success")
    return redirect(url_for("dashboard"))


# ---------------------------------------------------------
# Import & Export Routes
# ---------------------------------------------------------
@app.route("/import", methods=["GET", "POST"])
@login_required
def import_csv():
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

                # Anomaly check
                is_anom, reason = detect_transaction_anomaly(
                    session["user_id"], amount, category, tx_date, normalized["description"]
                )

                db.session.add(Transaction(
                    user_id=session["user_id"],
                    description=normalized["description"],
                    amount=amount,
                    tx_date=tx_date,
                    kind=kind,
                    category=category,
                    ai_category=predicted["category"],
                    ai_confidence=predicted["confidence"],
                    is_anomaly=is_anom,
                    anomaly_reason=reason
                ))
                count += 1

            db.session.commit()
            flash(f"Successfully imported and AI-categorized {count} transactions.", "success")
            return redirect(url_for("transactions"))
        except Exception as exc:
            db.session.rollback()
            flash(f"Import failed: {exc}", "danger")

    return render_template("import.html")


@app.route("/export/csv")
@login_required
def export_csv():
    rows = Transaction.query.filter_by(user_id=session["user_id"]).order_by(Transaction.tx_date.desc()).all()
    stream = BytesIO()
    import io
    text_stream = io.StringIO()
    writer = csv.writer(text_stream)
    writer.writerow([
        "date", "description", "amount", "type", "category",
        "ai_category", "ai_confidence", "is_anomaly", "anomaly_reason", "payment_method"
    ])
    for t in rows:
        writer.writerow([
            t.tx_date.isoformat(),
            t.description,
            f"{t.amount:.2f}",
            t.kind,
            t.category,
            t.ai_category,
            f"{t.ai_confidence:.2f}",
            "Yes" if t.is_anomaly else "No",
            t.anomaly_reason or "",
            t.payment_method
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
    txs = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.tx_date.desc()).all()
    health = calculate_financial_health(user.id)
    forecast = forecast_monthly_spend(user.id)
    pdf_buffer = generate_pdf_statement(user, txs, health, forecast)
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"spendsense_financial_statement_{date.today().strftime('%Y%m%d')}.pdf"
    )


# ---------------------------------------------------------
# REST APIs for Dynamic UI & AI Copilot
# ---------------------------------------------------------
@app.route("/api/categorize", methods=["POST"])
@login_required
def api_categorize():
    """Live interactive classification API for instant badges on keystroke."""
    data = request.get_json(silent=True) or {}
    text = data.get("description", "")
    result = categorize_expense(text)
    return jsonify(result)


@app.route("/api/forecast")
@login_required
def api_forecast():
    return jsonify({
        "forecast": forecast_monthly_spend(session["user_id"]),
        "insights": summarize_spending(session["user_id"]),
        "health": calculate_financial_health(session["user_id"])
    })


@app.route("/api/copilot", methods=["POST"])
@login_required
def api_copilot():
    """AI financial chatbot endpoint."""
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"reply": "Please ask me a financial question."})
    reply = chat_copilot(session["user_id"], message)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
