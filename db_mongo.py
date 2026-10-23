"""MongoDB Database Layer for SpendSense AI.

Connects to MongoDB (Local or MongoDB Atlas Cloud) via pymongo.
Includes automated fallback to in-memory mongomock for seamless offline testing.
"""

import os
from datetime import datetime, date
from bson import ObjectId
import pymongo

_client = None
_db = None
_is_mock = False

def init_mongo():
    global _client, _db, _is_mock
    if _db is not None:
        return _db

    mongo_uri = os.environ.get("MONGO_URI", "").strip()
    
    # If MONGO_URI is set, try connecting to real MongoDB (e.g. MongoDB Atlas or local)
    if mongo_uri:
        try:
            client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=4000)
            client.admin.command('ping')
            _client = client
            # Extract database name from URI or default to spendsense_ai
            try:
                db_name = pymongo.uri_parser.parse_uri(mongo_uri).get('database') or "spendsense_ai"
            except Exception:
                db_name = "spendsense_ai"
            _db = _client[db_name]
            _is_mock = False
            print(f"[*] Connected to MongoDB: {db_name}")
            _create_indexes(_db)
            return _db
        except Exception as e:
            print(f"[!] Warning: Could not connect to MONGO_URI ({e}). Using mongomock.")

    # Try default local mongodb://localhost:27017
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/spendsense_ai", serverSelectionTimeoutMS=1500)
        client.admin.command('ping')
        _client = client
        _db = _client["spendsense_ai"]
        _is_mock = False
        print("[*] Connected to local MongoDB server.")
        _create_indexes(_db)
        return _db
    except Exception:
        pass

    # Seamless in-memory fallback for local environments without MongoDB installed
    import mongomock
    _client = mongomock.MongoClient()
    _db = _client["spendsense_ai"]
    _is_mock = True
    print("[*] MongoDB initialized using in-memory engine (set MONGO_URI to connect MongoDB Atlas).")
    return _db

def _create_indexes(db):
    try:
        db.users.create_index("email", unique=True)
        db.transactions.create_index([("user_id", 1), ("tx_date", -1)])
        db.budgets.create_index([("user_id", 1), ("month", 1)], unique=True)
        db.goals.create_index("user_id")
        db.subscriptions.create_index("user_id")
    except Exception:
        pass

def get_db():
    global _db
    if _db is None:
        return init_mongo()
    return _db

def is_mock_db():
    return _is_mock


# ---------------------------------------------------------
# Document Model Wrappers (Dot-Notation for Templates)
# ---------------------------------------------------------
class BaseDoc:
    def __init__(self, d):
        self._doc = d or {}

    @property
    def id(self):
        val = self._doc.get("_id")
        return str(val) if val else None

    def __getattr__(self, name):
        if name in self._doc:
            return self._doc[name]
        return None


class UserDoc(BaseDoc):
    @property
    def name(self):
        return self._doc.get("name", "User")

    @property
    def email(self):
        return self._doc.get("email", "")

    @property
    def password_hash(self):
        return self._doc.get("password_hash", "")

    @property
    def currency(self):
        return self._doc.get("currency", "₹")

    @property
    def monthly_income(self):
        return float(self._doc.get("monthly_income", 0.0))

    @property
    def created_at(self):
        ca = self._doc.get("created_at")
        if isinstance(ca, datetime):
            return ca
        return datetime.utcnow()


class TransactionDoc(BaseDoc):
    @property
    def user_id(self):
        return str(self._doc.get("user_id", ""))

    @property
    def description(self):
        return self._doc.get("description", "")

    @property
    def amount(self):
        return float(self._doc.get("amount", 0.0))

    @property
    def tx_date(self):
        d = self._doc.get("tx_date")
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        if isinstance(d, str):
            try:
                return datetime.strptime(d[:10], "%Y-%m-%d").date()
            except ValueError:
                pass
        return date.today()

    @property
    def kind(self):
        return self._doc.get("kind", "expense")

    @property
    def category(self):
        return self._doc.get("category", "Other")

    @property
    def ai_category(self):
        return self._doc.get("ai_category", "Other")

    @property
    def ai_confidence(self):
        return float(self._doc.get("ai_confidence", 0.0))

    @property
    def is_anomaly(self):
        return bool(self._doc.get("is_anomaly", False))

    @property
    def anomaly_reason(self):
        return self._doc.get("anomaly_reason")

    @property
    def payment_method(self):
        return self._doc.get("payment_method", "UPI / Card")

    @property
    def notes(self):
        return self._doc.get("notes")

    @property
    def created_at(self):
        return self._doc.get("created_at", datetime.utcnow())


class BudgetDoc(BaseDoc):
    @property
    def user_id(self):
        return str(self._doc.get("user_id", ""))

    @property
    def month(self):
        return self._doc.get("month", "")

    @property
    def amount(self):
        return float(self._doc.get("amount", 0.0))

    @property
    def category_limits(self):
        return self._doc.get("category_limits", {})


class GoalDoc(BaseDoc):
    @property
    def user_id(self):
        return str(self._doc.get("user_id", ""))

    @property
    def title(self):
        return self._doc.get("title", "")

    @property
    def target_amount(self):
        return float(self._doc.get("target_amount", 0.0))

    @property
    def current_amount(self):
        return float(self._doc.get("current_amount", 0.0))

    @property
    def target_date(self):
        d = self._doc.get("target_date")
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        if isinstance(d, str):
            try:
                return datetime.strptime(d[:10], "%Y-%m-%d").date()
            except ValueError:
                pass
        return date.today()

    @property
    def category(self):
        return self._doc.get("category", "General")

    @property
    def status(self):
        return self._doc.get("status", "active")

    @property
    def created_at(self):
        return self._doc.get("created_at", datetime.utcnow())


class SubscriptionDoc(BaseDoc):
    @property
    def user_id(self):
        return str(self._doc.get("user_id", ""))

    @property
    def name(self):
        return self._doc.get("name", "")

    @property
    def amount(self):
        return float(self._doc.get("amount", 0.0))

    @property
    def billing_cycle(self):
        return self._doc.get("billing_cycle", "monthly")

    @property
    def next_due_date(self):
        d = self._doc.get("next_due_date")
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        if isinstance(d, str):
            try:
                return datetime.strptime(d[:10], "%Y-%m-%d").date()
            except ValueError:
                pass
        return date.today()

    @property
    def category(self):
        return self._doc.get("category", "Entertainment")

    @property
    def is_active(self):
        return bool(self._doc.get("is_active", True))
