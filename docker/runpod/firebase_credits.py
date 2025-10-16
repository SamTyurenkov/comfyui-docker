"""
Firebase Credits Management Module
"""

import os
import json
from typing import Optional, Dict, Any
from datetime import datetime

# Firebase Admin SDK imports
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, auth as firebase_auth
    FIREBASE_AVAILABLE = True
except ImportError:
    print("firebase-admin not installed. Credit system will be disabled.")
    print("Install with: pip install firebase-admin")
    FIREBASE_AVAILABLE = False

# Configuration
CREDITS_COLLECTION = "users"
TRANSACTIONS_COLLECTION = "credit_transactions"
DEFAULT_CREDITS_FOR_NEW_USERS = 100

# Initialize Firebase Admin SDK
_firebase_initialized = False
_db = None


def initialize_firebase():
    """
    Initialize Firebase Admin SDK with credentials from environment or file.
    
    Environment variables:
    - FIREBASE_SERVICE_ACCOUNT_PATH: Path to service account JSON file
    
    Returns:
        bool: True if initialized successfully, False otherwise
    """
    global _firebase_initialized, _db
    
    if not FIREBASE_AVAILABLE:
        print("firebase_credits - Firebase Admin SDK not available")
        return False
    
    if _firebase_initialized:
        return True
    
    try:
        # Hardcoded path for RunPod volume
        service_account_path = "/runpod-volume/serverless-frontend-116ab-firebase-adminsdk-fbsvc-a924cab0e6.json"
        
        # Fallback to environment variable if hardcoded path doesn't exist
        if not os.path.exists(service_account_path):
            service_account_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH")
        
        if service_account_path and os.path.exists(service_account_path):
            print(f"firebase_credits - Initializing with service account from file: {service_account_path}")
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
            _db = firestore.client()
            _firebase_initialized = True
            print("firebase_credits - Firebase Admin SDK initialized successfully")
            return True
        
        # No credentials found
        print("firebase_credits - No Firebase credentials found. Set FIREBASE_SERVICE_ACCOUNT_PATH")
        print("firebase_credits - Credit system will be disabled")
        return False
        
    except Exception as e:
        print(f"firebase_credits - Error initializing Firebase: {e}")
        return False


def verify_user_token(id_token: str) -> Optional[Dict[str, Any]]:
    """
    Verify Firebase ID token from client and return decoded token with user info.
    
    Args:
        id_token: Firebase ID token from client
        
    Returns:
        Dict with user info if valid, None otherwise
    """
    if not _firebase_initialized:
        return None
    
    try:
        decoded_token = firebase_auth.verify_id_token(id_token)
        return {
            "uid": decoded_token.get("uid"),
            "email": decoded_token.get("email"),
        }
    except Exception as e:
        print(f"firebase_credits - Error verifying token: {e}")
        return None


def get_user_credits(user_id: str) -> Dict[str, Any]:
    """
    Get user's current credit balance.
    
    Args:
        user_id: Firebase user ID (uid)
        
    Returns:
        Dict with credits and error (if any)
    """
    if not _firebase_initialized:
        return {"credits": None, "error": "Firebase not initialized"}
    
    try:
        user_ref = _db.collection(CREDITS_COLLECTION).document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            # Create new user with default credits
            print(f"firebase_credits - Creating new user {user_id} with {DEFAULT_CREDITS_FOR_NEW_USERS} credits")
            user_ref.set({
                "credits": DEFAULT_CREDITS_FOR_NEW_USERS,
                "created_at": firestore.SERVER_TIMESTAMP,
            })
            return {"credits": DEFAULT_CREDITS_FOR_NEW_USERS, "error": None}
        
        user_data = user_doc.to_dict()
        credits = user_data.get("credits", 0)
        return {"credits": credits, "error": None}
        
    except Exception as e:
        print(f"firebase_credits - Error getting user credits: {e}")
        return {"credits": None, "error": str(e)}


def check_sufficient_credits(user_id: str, required_credits: float) -> Dict[str, Any]:
    """
    Check if user has sufficient credits for a job.
    
    Args:
        user_id: Firebase user ID
        required_credits: Minimum credits required
        
    Returns:
        Dict with sufficient (bool), current_credits, and error (if any)
    """
    result = get_user_credits(user_id)
    if result["error"]:
        return {"sufficient": False, "current_credits": 0, "error": result["error"]}
    
    current_credits = result["credits"]
    sufficient = current_credits >= required_credits
    
    return {
        "sufficient": sufficient,
        "current_credits": current_credits,
        "required_credits": required_credits,
        "error": None
    }


def deduct_credits(
    user_id: str,
    cost_info: Dict[str, Any],
    workflow_id: Optional[str] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deduct credits from user's balance after workflow execution.
    This is called server-side after the job completes, ensuring the user
    is charged even if they close the browser.
    
    Args:
        user_id: Firebase user ID
        cost_info: Cost information dict from calculate_job_cost()
        workflow_id: ID of the workflow that was executed
        job_id: RunPod job ID
        
    Returns:
        Dict with success, new_balance, cost, and error (if any)
    """
    if not _firebase_initialized:
        return {
            "success": False,
            "new_balance": None,
            "cost": 0,
            "error": "Firebase not initialized"
        }
    
    try:
        cost = cost_info["total_cost"]
        user_ref = _db.collection(CREDITS_COLLECTION).document(user_id)
        
        # Use a transaction to ensure atomic update
        @firestore.transactional
        def update_credits(transaction):
            user_doc = user_ref.get(transaction=transaction)
            
            if not user_doc.exists:
                raise ValueError(f"User {user_id} not found")
            
            current_credits = user_doc.to_dict().get("credits", 0)
            new_balance = current_credits - cost
            
            # Allow negative balance (will be handled by business logic)
            # Or uncomment to prevent negative balance:
            # if new_balance < 0:
            #     raise ValueError(f"Insufficient credits. Current: {current_credits}, Required: {cost}")
            
            # Update user credits
            transaction.update(user_ref, {
                "credits": new_balance,
                "last_transaction": firestore.SERVER_TIMESTAMP,
            })
            
            return new_balance
        
        # Execute transaction
        transaction = _db.transaction()
        new_balance = update_credits(transaction)
        
        # Log transaction for audit trail
        try:
            transaction_ref = _db.collection(TRANSACTIONS_COLLECTION).document()
            transaction_ref.set({
                "user_id": user_id,
                "type": "deduction",
                "amount": cost,
                "workflow_id": workflow_id,
                "job_id": job_id,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "cost_breakdown": {
                    "base_cost": cost_info.get("base_cost"),
                    "execution_cost": cost_info.get("execution_cost"),
                    "execution_time_sec": cost_info.get("execution_time_sec"),
                    "gpu_type": cost_info.get("gpu_type"),
                    "rate_per_second": cost_info.get("rate_per_second"),
                }
            })
            print(f"firebase_credits - Transaction logged for user {user_id}")
        except Exception as log_error:
            print(f"firebase_credits - Warning: Failed to log transaction: {log_error}")
        
        print(f"firebase_credits - Deducted {cost} credits from user {user_id}. New balance: {new_balance}")
        return {
            "success": True,
            "new_balance": new_balance,
            "cost": cost,
            "error": None
        }
        
    except Exception as e:
        print(f"firebase_credits - Error deducting credits: {e}")
        return {
            "success": False,
            "new_balance": None,
            "cost": 0,
            "error": str(e)
        }


def add_credits(user_id: str, amount: float, reason: str = "manual_addition") -> Dict[str, Any]:
    """
    Add credits to user's balance (for admin/payment processing).
    
    Args:
        user_id: Firebase user ID
        amount: Credits to add
        reason: Reason for adding credits
        
    Returns:
        Dict with success, new_balance, and error (if any)
    """
    if not _firebase_initialized:
        return {"success": False, "new_balance": None, "error": "Firebase not initialized"}
    
    try:
        user_ref = _db.collection(CREDITS_COLLECTION).document(user_id)
        
        @firestore.transactional
        def update_credits(transaction):
            user_doc = user_ref.get(transaction=transaction)
            
            if not user_doc.exists:
                # Create user if doesn't exist
                transaction.set(user_ref, {
                    "credits": amount,
                    "created_at": firestore.SERVER_TIMESTAMP,
                })
                return amount
            
            current_credits = user_doc.to_dict().get("credits", 0)
            new_balance = current_credits + amount
            
            transaction.update(user_ref, {
                "credits": new_balance,
                "last_transaction": firestore.SERVER_TIMESTAMP,
            })
            
            return new_balance
        
        transaction = _db.transaction()
        new_balance = update_credits(transaction)
        
        # Log transaction
        try:
            transaction_ref = _db.collection(TRANSACTIONS_COLLECTION).document()
            transaction_ref.set({
                "user_id": user_id,
                "type": "addition",
                "amount": amount,
                "reason": reason,
                "timestamp": firestore.SERVER_TIMESTAMP,
            })
        except Exception as log_error:
            print(f"firebase_credits - Warning: Failed to log transaction: {log_error}")
        
        print(f"firebase_credits - Added {amount} credits to user {user_id}. New balance: {new_balance}")
        return {"success": True, "new_balance": new_balance, "error": None}
        
    except Exception as e:
        print(f"firebase_credits - Error adding credits: {e}")
        return {"success": False, "new_balance": None, "error": str(e)}


# Initialize on module import
initialize_firebase()
