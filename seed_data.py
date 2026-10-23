"""1-Click Realistic 6-Month Demo Data Seeder for Viva Presentations and Evaluators."""
from datetime import date, timedelta
import random
from models import db, User, Transaction, Budget, SavingsGoal, Subscription
from ai_engine import categorize_expense, detect_transaction_anomaly
from werkzeug.security import generate_password_hash

def seed_demo_data_for_user(user_id):
    """Populates 6 months of realistic financial history, budgets, goals, and subscriptions."""
    # Clear existing transactions, budgets, goals, subscriptions for this user
    Transaction.query.filter_by(user_id=user_id).delete()
    Budget.query.filter_by(user_id=user_id).delete()
    SavingsGoal.query.filter_by(user_id=user_id).delete()
    Subscription.query.filter_by(user_id=user_id).delete()
    db.session.commit()

    today = date.today()

    # 1. Seed Budgets for past 6 months and current month
    for month_offset in range(6, -1, -1):
        target_date = today - timedelta(days=month_offset * 30)
        m_str = target_date.strftime("%Y-%m")
        budget_amt = 42000.0 if month_offset > 0 else 45000.0
        db.session.add(Budget(
            user_id=user_id,
            month=m_str,
            amount=budget_amt,
            category_limits='{"Food": 12000, "Housing": 18000, "Transport": 4000, "Shopping": 5000}'
        ))

    # 2. Seed Savings Goals
    db.session.add(SavingsGoal(
        user_id=user_id,
        title="Emergency Buffer Fund",
        target_amount=150000.0,
        current_amount=95000.0,
        target_date=today + timedelta(days=120),
        category="Emergency",
        status="active"
    ))
    db.session.add(SavingsGoal(
        user_id=user_id,
        title="New MacBook Pro / Dev Workstation",
        target_amount=120000.0,
        current_amount=48000.0,
        target_date=today + timedelta(days=180),
        category="Tech",
        status="active"
    ))

    # 3. Seed Recurring Subscriptions
    subs = [
        ("Netflix Premium 4K", 649.0, "monthly", today + timedelta(days=12), "Entertainment"),
        ("Spotify Family Plan", 179.0, "monthly", today + timedelta(days=5), "Entertainment"),
        ("Airtel Xstream Fiber 300Mbps", 999.0, "monthly", today + timedelta(days=18), "Utilities"),
        ("Gold's Gym Membership", 1800.0, "monthly", today + timedelta(days=22), "Healthcare"),
        ("Amazon Prime Annual", 1499.0, "yearly", today + timedelta(days=85), "Shopping")
    ]
    for name, amt, cycle, due, cat in subs:
        db.session.add(Subscription(
            user_id=user_id,
            name=name,
            amount=amt,
            billing_cycle=cycle,
            next_due_date=due,
            category=cat,
            is_active=True
        ))

    # 4. Monthly recurring template items
    recurring_expenses_per_month = [
        ("Monthly Apartment Rent to Landlord", 18000.0, "Housing"),
        ("Bescom Electricity Bill", 1650.0, "Utilities"),
        ("Broadband Internet Fiber Bill", 999.0, "Utilities"),
        ("Netflix Premium Subscription", 649.0, "Entertainment"),
        ("Spotify Premium Music", 179.0, "Entertainment"),
        ("Gym Monthly Membership", 1800.0, "Healthcare"),
    ]

    discretionary_pool = [
        ("Swiggy Biryani Dinner with Colleagues", 680.0, "Food"),
        ("Zomato Gourmet Lunch Delivery", 420.0, "Food"),
        ("Supermarket D-Mart Grocery Provisions", 3450.0, "Food"),
        ("Starbucks Cold Brew & Croissant", 390.0, "Food"),
        ("Uber Premier Ride to Tech Park", 340.0, "Transport"),
        ("Ola Auto Commute to Metro", 95.0, "Transport"),
        ("Petrol Refill Shell Fuel Station", 1500.0, "Transport"),
        ("Amazon Online Purchase Books & Gadgets", 1850.0, "Shopping"),
        ("Zara Casual Shirts Retail Mall", 2990.0, "Shopping"),
        ("PVR IMAX Movie Tickets & Popcorn", 1100.0, "Entertainment"),
        ("Apollo Pharmacy Medicines & Health Supplements", 850.0, "Healthcare"),
        ("Coursera Machine Learning Specialization", 2400.0, "Education")
    ]

    # Generate 6 months of historical transactions
    for m in range(6, -1, -1):
        base_month_date = today - timedelta(days=m * 30)
        
        # 1st of month: Salary Credit
        sal_date = base_month_date.replace(day=1)
        db.session.add(Transaction(
            user_id=user_id,
            description="Monthly Software Engineer Salary Credit",
            amount=75000.0,
            tx_date=sal_date,
            kind="income",
            category="Income",
            ai_category="Income",
            ai_confidence=0.98,
            payment_method="Direct Bank Deposit"
        ))

        # Fixed recurring expenses
        for desc, amt, cat in recurring_expenses_per_month:
            d_day = min(random.randint(2, 7), 28)
            tx_d = base_month_date.replace(day=d_day)
            pred = categorize_expense(desc)
            db.session.add(Transaction(
                user_id=user_id,
                description=desc,
                amount=amt,
                tx_date=tx_d,
                kind="expense",
                category=cat,
                ai_category=pred["category"],
                ai_confidence=pred["confidence"],
                is_recurring=True,
                payment_method="UPI / Auto-Debit"
            ))

        # Random discretionary transactions for realism
        num_random = random.randint(7, 12)
        for _ in range(num_random):
            item = random.choice(discretionary_pool)
            d_day = min(random.randint(5, 27), 28)
            tx_d = base_month_date.replace(day=d_day)
            # Add slight realistic noise to amount (+/- 10%)
            jitter = random.uniform(0.9, 1.15)
            actual_amt = round(item[1] * jitter, 2)
            pred = categorize_expense(item[0])
            db.session.add(Transaction(
                user_id=user_id,
                description=item[0],
                amount=actual_amt,
                tx_date=tx_d,
                kind="expense",
                category=item[2],
                ai_category=pred["category"],
                ai_confidence=pred["confidence"],
                is_recurring=False,
                payment_method="UPI / Card"
            ))

    # Add deliberate Anomalies for viva demonstration
    # 1. Anomaly 1: Sudden spike in Food
    recent_spike_date = today - timedelta(days=4)
    db.session.add(Transaction(
        user_id=user_id,
        description="Luxury 5-Star Hotel Banquet Dinner Feast",
        amount=14500.0,
        tx_date=recent_spike_date,
        kind="expense",
        category="Food",
        ai_category="Food",
        ai_confidence=0.92,
        is_anomaly=True,
        anomaly_reason="Spending Spike: 4.8x higher than typical Food expense (median: ₹550)",
        payment_method="Credit Card"
    ))

    # 2. Anomaly 2: Duplicate transaction charge
    dup_date = today - timedelta(days=2)
    db.session.add(Transaction(
        user_id=user_id,
        description="Electronics Mega Store Audio Monitor",
        amount=3890.0,
        tx_date=dup_date,
        kind="expense",
        category="Shopping",
        ai_category="Shopping",
        ai_confidence=0.89,
        payment_method="Card"
    ))
    db.session.add(Transaction(
        user_id=user_id,
        description="Electronics Mega Store Audio Monitor (POS Retry)",
        amount=3890.0,
        tx_date=dup_date,
        kind="expense",
        category="Shopping",
        ai_category="Shopping",
        ai_confidence=0.89,
        is_anomaly=True,
        anomaly_reason="Potential duplicate: identical amount ₹3890.00 logged within 48h",
        payment_method="Card"
    ))

    db.session.commit()
    return True

def ensure_demo_user():
    """Creates a standard demonstration account if not already present."""
    demo_email = "demo@spendsense.ai"
    user = User.query.filter_by(email=demo_email).first()
    if not user:
        user = User(
            name="Viva Examiner / Demo User",
            email=demo_email,
            password_hash=generate_password_hash("Demo1234!"),
            currency="₹",
            monthly_income=75000.0
        )
        db.session.add(user)
        db.session.commit()
        seed_demo_data_for_user(user.id)
    return user
