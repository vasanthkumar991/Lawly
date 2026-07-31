# authentication.py
import os
import json
import bcrypt
from typing import Dict, Tuple, Optional

USERS_FILE = "./user_data/users.json"
os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)

def load_users() -> Dict[str, Dict]:
    """Load users from local JSON file (returns empty dict on error)."""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}
    return {}

def save_users(users: Dict[str, Dict]) -> None:
    """Save users mapping to disk."""
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def normalize_email(email: str) -> str:
    """Trim and lowercase an email for consistent keys."""
    return (email or "").strip().lower()

def _find_key_for_email(users: Dict[str, Dict], email_norm: str) -> Optional[str]:
    """Return an existing key whose normalized form equals email_norm, or None."""
    for k in users.keys():
        if normalize_email(k) == email_norm:
            return k
    return None

def create_user(email: str, password: str, name: str = "") -> bool:
    """
    Create a new user with bcrypt-hashed password.
    Returns False if user already exists.
    """
    email_norm = normalize_email(email)
    users = load_users()
    if _find_key_for_email(users, email_norm):
        return False
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    users[email_norm] = {"password_hash": hashed_pw, "name": name}
    save_users(users)
    return True

def verify_user(email: str, password: str) -> Tuple[bool, str]:
    """
    Verify credentials. Returns (ok, display_name).
    Supports legacy plaintext password entries as a temporary fallback.
    """
    email_norm = normalize_email(email)
    users = load_users()

    rec = users.get(email_norm)
    if not rec:
        # fallback: find any legacy key that normalizes to this email
        fallback_key = _find_key_for_email(users, email_norm)
        if fallback_key:
            rec = users.get(fallback_key)

    if not rec:
        return False, ""

    stored = rec.get("password_hash") or rec.get("password") or ""
    if isinstance(stored, str):
        stored = stored.strip()
    else:
        stored = str(stored)

    # If stored value looks like a bcrypt hash (starts with $2), verify with bcrypt.
    if stored.startswith("$2"):
        try:
            ok = bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
        except Exception:
            ok = False
    else:
        # compatibility: plain-text comparison (only temporary - migrate to hashes)
        ok = (stored == password)

    if ok:
        return True, rec.get("name") or email_norm
    return False, ""

def migrate_normalize_keys(save: bool = True) -> Dict[str, Dict]:
    """
    One-time migration: normalize all keys to lowercase and convert plaintext passwords to hashes.
    - If an entry has 'password' (plaintext), it will be hashed and stored as 'password_hash'.
    - Returns the migrated dict and optionally writes it back to USERS_FILE when save=True.
    """
    users = load_users()
    new_users: Dict[str, Dict] = {}
    conflicts = []

    for orig_key, rec in users.items():
        norm = normalize_email(orig_key)
        password_plain = rec.get("password")
        password_hash = rec.get("password_hash")
        name = rec.get("name", "")

        if password_hash and isinstance(password_hash, str):
            ph = password_hash.strip()
        elif password_plain:
            # convert plaintext to bcrypt hash
            ph = bcrypt.hashpw(str(password_plain).encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        else:
            ph = ""

        entry = {"password_hash": ph, "name": name}

        if norm not in new_users:
            new_users[norm] = entry
        else:
            # collision: keep existing and log conflict
            if new_users[norm]["password_hash"] != entry["password_hash"]:
                conflicts.append((orig_key, norm))

    if save:
        save_users(new_users)
        if conflicts:
            log_path = os.path.splitext(USERS_FILE)[0] + "_migrate_conflicts.txt"
            with open(log_path, "w", encoding="utf-8") as f:
                for orig, norm in conflicts:
                    f.write(f"Conflict: {orig} -> {norm}\n")

    return new_users
