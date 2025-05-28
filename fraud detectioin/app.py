from flask import Flask, render_template, request, redirect, url_for, flash, session
import json
import os
import pickle
import numpy as np
from datetime import datetime
from functools import wraps
import uuid
import pandas as pd

app = Flask(__name__)
app.secret_key =  os.urandom(24)

# Load ML model
try:
    model = pickle.load(open("fraud_detection_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a placeholder model for development purposes
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier()
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

# Database files
USERS_DB = "users.json"
TRANSACTIONS_DB = "transactions.json"

# Add this route to your app.py file


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to access this page", "danger")
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


@app.route("/fraud_alerts")
@login_required
def fraud_alerts():
    user_id = session["user_id"]

    # Load all transactions
    transactions = load_transactions()

    # If transactions is a dictionary, convert it to a list
    if isinstance(transactions, dict):
        transactions = [transactions]

    # Filter transactions relevant to the current user
    user_transactions = [
        t for t in transactions if t["sender"] == user_id or t["receiver"] == user_id
    ]

    return render_template("fraud_alerts.html", transactions=user_transactions)

# Add this route to your app.py file after the transfer function


@app.route("/transfer_confirmation")
@login_required
def transfer_confirmation():
    """
    Display a confirmation page with fraud detection visualization
    This is called after a transfer is processed
    """
    # Get transaction details from query parameters
    transaction_id = request.args.get("transaction_id", "")
    is_fraud = request.args.get("is_fraud", "false") == "true"

    # Get transaction details
    transactions = load_transactions()

    # If transactions is a dictionary, convert it to a list
    if isinstance(transactions, dict):
        transactions = [transactions]

    # Find the specific transaction
    transaction = None
    for t in transactions:
        if t.get("id") == transaction_id:
            transaction = t
            break

    # Default values if transaction not found
    if not transaction:
        # Default values for demonstration
        transaction = {
            "sender": session.get("user_id", "unknown"),
            "receiver": request.args.get("receiver_id", "unknown"),
            "amount": float(request.args.get("amount", 0)),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": len(transactions) + 1,
            "oldbalanceOrg": float(request.args.get("oldbalanceOrg", 1000)),
            "newbalanceOrig": float(request.args.get("newbalanceOrig", 900)),
            "oldbalanceDest": float(request.args.get("oldbalanceDest", 500)),
            "newbalanceDest": float(request.args.get("newbalanceDest", 600)),
            "predicted_fraud": is_fraud,
        }

    return render_template("transfer_confirmation.html", **transaction)


# Initialize databases if they don't exist
def initialize_db():
    if not os.path.exists(USERS_DB):
        with open(USERS_DB, "w") as f:
            json.dump({}, f)

    if not os.path.exists(TRANSACTIONS_DB):
        with open(TRANSACTIONS_DB, "w") as f:
            json.dump([], f)


initialize_db()


# Helper functions
def load_users():
    with open(USERS_DB, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_DB, "w") as f:
        json.dump(users, f, indent=4)


def load_transactions():
    try:
        with open(TRANSACTIONS_DB, "r") as f:
            data = json.load(f)
            # Ensure we return a list
            if isinstance(data, list):
                return data
            else:
                # If it's not a list, initialize an empty list
                return []
    except (FileNotFoundError, json.JSONDecodeError):
        # Handle case where file doesn't exist or is invalid
        return []


def save_transactions(transactions):
    with open(TRANSACTIONS_DB, "w") as f:
        json.dump(transactions, f, indent=4)


# Login required decorator

# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/test_fraud_detection", methods=["GET", "POST"])
@login_required
def test_fraud_detection():
    if request.method == "POST":
        # Create a transaction with characteristics that would trigger fraud detection
        sender_id = session["user_id"]
        receiver_id = request.form["receiver_id"]
        amount = float(request.form["amount"])

        users = load_users()

        # Simulate transaction data that would trigger the ML model
        step = len(load_transactions()) + 1
        oldbalanceOrg = users[sender_id]["balance"]

        # Create unusual values likely to trigger fraud detection
        # Option 1: Very low remaining balance
        newbalanceOrig = -100  # Negative balance (impossible in normal transfers)

        # Option 2: Unusually large amount for receiver
        oldbalanceDest = users[receiver_id]["balance"]
        newbalanceDest = oldbalanceDest + amount * 10  # Unusual increase

        # Force the model to predict fraud
        is_fraud = True

        # Create the transaction record
        transaction = {
            "id": str(uuid.uuid4()),
            "sender": sender_id,
            "receiver": receiver_id,
            "amount": amount,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "predicted_fraud": is_fraud,
            "status": "blocked",
        }

        # Save the test transaction
        transactions = load_transactions()
        if isinstance(transactions, dict):
            transactions = [transactions]
        transactions.append(transaction)
        save_transactions(transactions)

        return redirect(
            url_for(
                "transfer_confirmation",
                transaction_id=transaction["id"],
                is_fraud="true",
                oldbalanceOrg=oldbalanceOrg,
                newbalanceOrig=newbalanceOrig,
                oldbalanceDest=oldbalanceDest,
                newbalanceDest=newbalanceDest,
            )
        )

    # GET request - show the test form
    users = load_users()
    return render_template("test_fraud.html", users=users)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        users = load_users()

        user_id = request.form["user_id"]
        name = request.form["name"]
        password = request.form["password"]
        pin = request.form["pin"]

        if user_id in users:
            flash("User ID already exists", "danger")
            return redirect(url_for("register"))

        # Create new user
        users[user_id] = {
            "name": name,
            "password": password,  # In a real app, hash this password
            "pin": pin,  # In a real app, hash this PIN
            "balance": 10000.0,  # Initial balance
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        save_users(users)
        flash("Registration successful! You can now login", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        users = load_users()

        user_id = request.form["user_id"]
        password = request.form["password"]

        if user_id in users and users[user_id]["password"] == password:
            session["user_id"] = user_id
            session["name"] = users[user_id]["name"]
            flash(f'Welcome back, {users[user_id]["name"]}!', "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    users = load_users()
    user_id = session["user_id"]
    user = users[user_id]

    # Get user's transactions
    transactions = load_transactions()
    user_transactions = [
        t for t in transactions if t["sender"] == user_id or t["receiver"] == user_id
    ]

    return render_template("dashboard.html", user=user, transactions=user_transactions)


@app.route("/transfer", methods=["GET", "POST"])
@login_required
def transfer():
    users = load_users()
    sender_id = session["user_id"]

    if request.method == "POST":
        receiver_id = request.form["receiver_id"]
        amount = float(request.form["amount"])
        pin = request.form["pin"]

        # Validate inputs
        if receiver_id not in users:
            flash("Receiver does not exist", "danger")
            return redirect(url_for("transfer"))

        if sender_id == receiver_id:
            flash("Cannot transfer to yourself", "danger")
            return redirect(url_for("transfer"))

        if users[sender_id]["pin"] != pin:
            flash("Invalid PIN", "danger")
            return redirect(url_for("transfer"))

        if amount <= 0:
            flash("Amount must be positive", "danger")
            return redirect(url_for("transfer"))

        if users[sender_id]["balance"] < amount:
            flash("Insufficient balance", "danger")
            return redirect(url_for("transfer"))

        # Get current values for fraud detection
        step = len(load_transactions()) + 1  # transaction step
        oldbalanceOrg = users[sender_id]["balance"]
        newbalanceOrig = oldbalanceOrg - amount
        oldbalanceDest = users[receiver_id]["balance"]
        newbalanceDest = oldbalanceDest + amount

        # Predict fraud
        features = np.array(
            [[step, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]]
        )
        features_scaled = scaler.transform(features)
        is_fraud = model.predict(features_scaled)[0]

        transaction = {
            "id": str(uuid.uuid4()),
            "sender": sender_id,
            "receiver": receiver_id,
            "amount": amount,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "predicted_fraud": bool(is_fraud),
        }

        # Process the transaction
        if is_fraud:
            flash(
                "This transaction seems suspicious and has been flagged as potential fraud. Transaction cancelled.",
                "danger",
            )

            # Save the attempted transaction with fraud flag
            transactions = load_transactions()
            if isinstance(transactions, dict):
                transactions = [transactions]

            transaction["status"] = "blocked"
            transactions.append(transaction)
            save_transactions(transactions)

            # Redirect to confirmation page showing the fraud detection analysis
            return redirect(
                url_for(
                    "transfer_confirmation",
                    transaction_id=transaction["id"],
                    is_fraud="true",
                    oldbalanceOrg=oldbalanceOrg,
                    newbalanceOrig=newbalanceOrig,
                    oldbalanceDest=oldbalanceDest,
                    newbalanceDest=newbalanceDest,
                )
            )

        # Process the transaction
        users[sender_id]["balance"] -= amount
        users[receiver_id]["balance"] += amount

        # Save updated user data
        save_users(users)

        # Save transaction record
        transactions = load_transactions()
        if isinstance(transactions, dict):
            transactions = [transactions]

        transaction["status"] = "completed"
        transactions.append(transaction)
        save_transactions(transactions)

        flash(
            f'Successfully transferred ₹{amount} to {users[receiver_id]["name"]}',
            "success",
        )

        # Redirect to confirmation page
        return redirect(
            url_for(
                "transfer_confirmation",
                transaction_id=transaction["id"],
                is_fraud="false",
                oldbalanceOrg=oldbalanceOrg,
                newbalanceOrig=newbalanceOrig,
                oldbalanceDest=oldbalanceDest,
                newbalanceDest=newbalanceDest,
            )
        )

    return render_template("transfer.html", users=users)


def predict_fraud(features, model, scaler):
    # For testing: Detect as fraud if the amount is exactly 1337
    if features[2] == -1337:  # If newbalanceOrig is -1337
        return True

    # Normal detection logic
    features = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    return bool(prediction)


@app.route("/transactions")
@login_required
def transactions():
    user_id = session["user_id"]
    transactions = load_transactions()
    user_transactions = [
        t for t in transactions if t["sender"] == user_id or t["receiver"] == user_id
    ]

    return render_template("transactions.html", transactions=user_transactions)


if __name__ == "__main__":
    app.run(debug=True)
