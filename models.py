from datetime import datetime, date
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    currency = db.Column(db.String(10), default="₹", nullable=False)
    monthly_income = db.Column(db.Float, default=0.0, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    transactions = db.relationship("Transaction", backref="user", cascade="all, delete-orphan", lazy="dynamic")
    budgets = db.relationship("Budget", backref="user", cascade="all, delete-orphan", lazy="dynamic")
    goals = db.relationship("SavingsGoal", backref="user", cascade="all, delete-orphan", lazy="dynamic")
    subscriptions = db.relationship("Subscription", backref="user", cascade="all, delete-orphan", lazy="dynamic")


class Transaction(db.Model):
    __tablename__ = "transaction"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    description = db.Column(db.String(255), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    tx_date = db.Column(db.Date, nullable=False, index=True)
    kind = db.Column(db.String(10), nullable=False, default="expense") # expense | income
    category = db.Column(db.String(40), nullable=False, default="Other")
    ai_category = db.Column(db.String(40), nullable=False, default="Other")
    ai_confidence = db.Column(db.Float, nullable=False, default=0.0)
    
    # Advanced AI features: Anomaly & Recurring
    is_anomaly = db.Column(db.Boolean, default=False, nullable=False)
    anomaly_reason = db.Column(db.String(255), nullable=True)
    is_recurring = db.Column(db.Boolean, default=False, nullable=False)
    payment_method = db.Column(db.String(40), default="UPI / Card", nullable=False)
    receipt_data = db.Column(db.Text, nullable=True) # JSON or extracted OCR notes
    notes = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class Budget(db.Model):
    __tablename__ = "budget"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    month = db.Column(db.String(7), nullable=False, index=True) # YYYY-MM
    amount = db.Column(db.Float, nullable=False)
    # JSON dictionary of category limits, e.g. {"Food": 5000, "Shopping": 3000}
    category_limits = db.Column(db.Text, default="{}", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (db.UniqueConstraint("user_id", "month", name="uq_user_month_budget"),)


class SavingsGoal(db.Model):
    __tablename__ = "savings_goal"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    title = db.Column(db.String(120), nullable=False)
    target_amount = db.Column(db.Float, nullable=False)
    current_amount = db.Column(db.Float, default=0.0, nullable=False)
    target_date = db.Column(db.Date, nullable=False)
    category = db.Column(db.String(50), default="General", nullable=False)
    status = db.Column(db.String(20), default="active", nullable=False) # active, completed
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class Subscription(db.Model):
    __tablename__ = "subscription"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    billing_cycle = db.Column(db.String(20), default="monthly", nullable=False) # monthly | yearly
    next_due_date = db.Column(db.Date, nullable=False)
    category = db.Column(db.String(40), default="Entertainment", nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
