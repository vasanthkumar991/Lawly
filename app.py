import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path
import html
import re
import os
import json
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
for env_path in [BASE_DIR / ".env", BASE_DIR / "scripts" / ".env"]:
    if env_path.exists():
        load_dotenv(env_path, override=False)

# Import bcrypt-safe auth helpers from authentication.py
from authentication import create_user, verify_user, load_users, save_users

# -----------------------
# CONFIG - keep secrets in local environment only
# -----------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
INDEX_NAME = "legal-cases"
TOP_K = 4

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError(
        "Missing GEMINI_API_KEY and/or PINECONE_API_KEY environment variables. "
        "Set them in your shell or create a local .env file in the project root."
    )

# Data dir for per-user persistence (local dev)
DATA_DIR = "./user_data"
os.makedirs(DATA_DIR, exist_ok=True)
USERS_FILE = os.path.join(DATA_DIR, "users.json")

# -----------------------
# Helpers: Gemini / Pinecone
# -----------------------
def init_clients(gemini_api_key: str, pinecone_api_key: str, pinecone_env: str):
    genai.configure(api_key=gemini_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    return genai, pc


def embed_text_with_gemini(query_text: str):
    response = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query_text,
        task_type="retrieval_query",
    )
    if isinstance(response, dict):
        embedding = response.get("embedding")
        if embedding is not None:
            return embedding
    embedding = getattr(response, "embedding", None)
    if embedding is not None:
        return embedding
    raise ValueError("Could not extract embedding from Gemini response.")


def pinecone_query(openai_client, pc: Pinecone, index_name: str, query_text: str, top_k: int = 4):
    index = pc.Index(index_name)
    query_vector = embed_text_with_gemini(query_text)
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    hits = []
    for match in result.get("matches", []):
        meta = match.get("metadata", {}) or {}
        text = meta.get("text") or meta.get("content") or ""
        hits.append({
            "id": match.get("id"),
            "score": match.get("score", 0),
            "metadata": meta,
            "text": text,
        })
    return hits


def build_rag_prompt(user_question: str, retrieved: List[Dict], instructions: str = "You are a helpful legal assistant.") -> str:
    context_parts = []
    for r in retrieved:
        snippet = r.get("text", "")
        src = r.get("metadata", {}).get("source", r.get("id"))
        snippet_short = snippet[:800].rsplit('\n', 1)[0] if snippet else ""
        context_parts.append(f"[SOURCE:{src}] {snippet_short}")
    context = "\n\n".join(context_parts)
    prompt = (
        f"{instructions}\n\n"
        "Use the following extracted sections from legal documents and cite them inline by source id "
        "(e.g. [SOURCE:doc_123]).\n\n"
        f"Context:\n{context}\n\n"
        f"User Question:\n{user_question}\n\n"
        "Answer succinctly and clearly. If you can't answer from the provided sources, say so and avoid hallucination."
    )
    return prompt


def query_openai_chat(openai_client, prompt: str, temperature: float = 0.0, max_tokens: int = 800) -> str:
    model = openai_client.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    text = getattr(response, "text", "")
    if not text:
        try:
            text = response.candidates[0].content.parts[0].text
        except Exception:
            text = str(response)
    return text.strip()

def make_chat_title(messages: List[Dict]) -> str:
    if not messages:
        return "New Chat"
    for m in messages:
        if m.get("role") == "user" and m.get("text"):
            raw = m["text"].strip()
            title = (raw[:40] + "...") if len(raw) > 40 else raw
            return title
    return datetime.now().strftime("Chat %Y-%m-%d %H:%M")

# -----------------------
# Simple local user store for demo only (do not use in production)
# -----------------------
DEMO_USER = {"email": "demo@lawly.local", "password": "password123", "name": "Demo User"}
EMAIL_REGEX = r"^[\w\.-]+@[\w\.-]+\.\w+$"

# -----------------------
# Persistence helpers (simple JSON per-user)
# -----------------------
def user_file_path(email: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.@-]", "_", email or "unknown")
    return os.path.join(DATA_DIR, f"{safe}.json")

def load_user_history(email: str) -> List[Dict]:
    path = user_file_path(email)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("history", [])
        except Exception:
            return []
    return []

def save_user_history(email: str, history: List[Dict]):
    path = user_file_path(email)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"history": history}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # ignore save errors in demo

# -----------------------
# App init & session state
# -----------------------
st.set_page_config(page_title="Lawly", layout="wide")

# session state defaults
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "is_guest" not in st.session_state:
    st.session_state.is_guest = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

# chat state (will be initialized after login)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_history_idx" not in st.session_state:
    st.session_state.selected_history_idx = None
if "nav_search" not in st.session_state:
    st.session_state.nav_search = ""
# Custom white + navy blue theme for Lawly

# -----------------------
# Authentication helpers
# -----------------------
def logout_all():
    # Save history for non-guest users before clearing
    if st.session_state.get("logged_in") and not st.session_state.get("is_guest") and st.session_state.get("user_email"):
        save_user_history(st.session_state.user_email, st.session_state.get("history", []))
    keys_to_clear = ["logged_in", "is_guest", "user_email", "user_name",
                     "gemini_client", "pc", "messages", "history",
                     "selected_history_idx", "nav_search", "show_signup", "page"]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# dev reset in sidebar
if st.sidebar.button("Reset session (dev)"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# -----------------------
# LOGIN / GUEST / SIGNUP PAGE (centered and dark-friendly)
# -----------------------
def show_login_page():
    st.markdown("<br/>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown('<h2 style="text-align:center">⚖️ Lawly</h2>', unsafe_allow_html=True)
        st.markdown("### Sign in to Lawly")
        st.write("Access your saved chats or continue as a guest.")

        # form begins
        with st.form(key="login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")

            if submitted:
                # --- DEBUG: inspect stored record & direct bcrypt test (optional) ---
                # Remove or comment this block once login is verified working.
                try:
                    import bcrypt
                    email_norm = (email or "").strip().lower()
                    users_debug = load_users()

                    # lookup using normalized key or fallback
                    rec = users_debug.get(email_norm)
                    if not rec:
                        for k, v in users_debug.items():
                            if (k or "").strip().lower() == email_norm:
                                rec = v
                                break

                    st.write("Keys in users.json:", list(users_debug.keys()))
                    st.write("Found record for given email key:", email_norm in users_debug)
                    st.write("Fallback key:", next((k for k in users_debug.keys() if (k or '').strip().lower()==email_norm), None))
                    st.write("Stored record (raw):", rec)
                    stored_raw = (rec.get("password_hash") if rec else None) or (rec.get("password") if rec else None)
                    st.write("Stored raw repr:", repr(stored_raw))
                    st.write("Type of stored_raw:", type(stored_raw))

                    if stored_raw:
                        try:
                            bcrypt_ok = bcrypt.checkpw(password.encode("utf-8"), str(stored_raw).encode("utf-8"))
                        except Exception as e:
                            bcrypt_ok = f"bcrypt raised: {e}"
                        st.write("bcrypt.checkpw result:", bcrypt_ok)
                    else:
                        st.write("No stored password/hash found for this user.")
                except Exception as _e:
                    # debug should not break flow
                    st.write("Debug check raised:", _e)
                # --- end debug ---

                # Use the imported verify_user (from authentication.py)
                ok, name = verify_user(email.strip(), password)
                if ok:
                    st.success(f"Welcome back, {name or email}")
                    st.session_state.logged_in = True
                    st.session_state.is_guest = False
                    st.session_state.user_email = (email or "").strip().lower()
                    st.session_state.user_name = name
                    st.rerun()
                else:
                    st.error("Invalid email or password")
        # form ends

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Continue as Guest"):
                st.session_state.logged_in = True
                st.session_state.is_guest = True
                st.session_state.user_email = "guest"
                st.session_state.user_name = "Guest"
                st.rerun()
        with col2:
            if st.button("Sign up"):
                st.session_state.page = "signup"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

def show_signup_page():
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("## Create a Lawly account")
    st.write("Sign up to save your chats securely.")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.form(key="signup_form"):
            email = st.text_input("Email")
            name = st.text_input("Display name (optional)")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Create account")

            if submitted:
                if not re.match(EMAIL_REGEX, email or ""):
                    st.error("Enter a valid email address.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif len(password or "") < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    ok = create_user(email.strip(), password, name.strip())
                    if ok:
                        st.success("Account created successfully! Please sign in.")
                        st.session_state.page = "login"
                        st.rerun()
                    else:
                        st.warning("User already exists.")

        back_col1, back_col2 = st.columns(2)
        with back_col1:
            if st.button("← Back to Sign in"):
                st.session_state.page = "login"
                st.rerun()
        with back_col2:
            if st.button("Continue as Guest"):
                st.session_state.logged_in = True
                st.session_state.is_guest = True
                st.session_state.user_email = "guest"
                st.session_state.user_name = "Guest"
                st.rerun()
    st.stop()

# -----------------------
# LOGIN CHECK - show login page if user not logged in
# -----------------------
# initialize page selector if missing
if "page" not in st.session_state:
    st.session_state.page = "login"

# If user not logged in, render pages (login or signup)
if not st.session_state.get("logged_in", False):
    # show signup page if requested, otherwise login page
    if st.session_state.get("page") == "signup":
        show_signup_page()
    else:
        # default to login
        st.session_state.page = "login"
        show_login_page()

# -----------------------
# Initialize Gemini / Pinecone clients after login
# -----------------------
try:
    if "gemini_client" not in st.session_state or "pc" not in st.session_state:
        gemini_client, pc = init_clients(GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_ENV)
        st.session_state.gemini_client = gemini_client
        st.session_state.pc = pc
except Exception as e:
    st.sidebar.error(f"Error initializing API clients: {e}")

# -----------------------
# MAIN APP (User is logged in or guest)
# -----------------------

# top header (safe user display) with logout button on the right
user_display = st.session_state.get("user_name") or st.session_state.get("user_email") or ""
user_display_escaped = html.escape(str(user_display))

# header layout: title left, logout right
hcol1, hcol2 = st.columns([0.85, 0.15])
with hcol1:
    st.markdown(
        "<div style='display:flex; align-items:center; gap:12px; padding:6px 0;'>"
        "<div style='font-size:28px;'>⚖️</div>"
        "<div><h1 style='margin:0'>Lawly</h1>"
        f"<div style='color:var(--text-secondary); font-size:14px'>Welcome, {user_display_escaped}</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
with hcol2:
    # show Logout button in header
    if st.button("Logout", key="header_logout"):
        logout_all()

st.markdown("---")

# Minimal sidebar info
st.sidebar.header("Lawly")
if st.session_state.is_guest:
    st.sidebar.caption("Signed in as: Guest (limited)")
else:
    st.sidebar.caption(f"Signed in as: {st.session_state.user_email}")
st.sidebar.markdown("---")
st.sidebar.caption("Lawly is for research only. Do not input confidential data. Consult a qualified lawyer for legal decisions.")

# Page layout: left nav + main
left_fraction = 0.22
nav_col, main_col = st.columns([left_fraction, 1 - left_fraction])

# ---------- LEFT NAV ----------
with nav_col:
    # New Chat
    if st.button("➕  New chat", key="nav_new", use_container_width=True):
        if st.session_state.messages:
            title = make_chat_title(st.session_state.messages)
            st.session_state.history.insert(0, {
                "title": title,
                "messages": st.session_state.messages.copy(),
                "created": datetime.now().isoformat()
            })
        st.session_state.messages = []
        st.session_state.selected_history_idx = None
        st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Search box
    search_input = st.text_input("Search chats", value=st.session_state.get("nav_search", ""), key="nav_search_input")
    st.session_state.nav_search = search_input or ""
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Scrollable chat list
    st.markdown("<div style='max-height:520px; overflow:auto; padding-right:6px;'>", unsafe_allow_html=True)

    query = st.session_state.nav_search.lower().strip()
    filtered = []
    for idx, item in enumerate(st.session_state.history):
        if not query or query in item["title"].lower():
            filtered.append((idx, item))

    if not filtered:
        st.markdown("<div style='opacity:0.7; padding:8px;'>No chats yet. Create a new chat to start.</div>", unsafe_allow_html=True)
    else:
        for idx, hist in filtered:
            is_selected = (st.session_state.selected_history_idx == idx)
            bg = "" if is_selected else "transparent"
            border = "1px solid rgba(255,255,255,0.03)" if is_selected else "none"
            st.markdown(f"<div style='background:{bg}; border:{border}; padding:8px; border-radius:8px; margin-bottom:6px;'>", unsafe_allow_html=True)
            cols = st.columns([5,1])
            with cols[0]:
                if st.button(hist["title"], key=f"load_title_{idx}", help="Load this chat"):
                    st.session_state.messages = hist["messages"].copy()
                    st.session_state.selected_history_idx = idx
                    item = st.session_state.history.pop(idx)
                    st.session_state.history.insert(0, item)
                    st.session_state.selected_history_idx = 0
                    st.rerun()
                st.markdown(f"<small style='opacity:0.6'>{hist.get('created','')}</small>", unsafe_allow_html=True)
            with cols[1]:
                if st.button("🗑️", key=f"delete_{idx}", help="Delete this chat"):
                    st.session_state.history.pop(idx)
                    if st.session_state.selected_history_idx == idx:
                        st.session_state.messages = []
                        st.session_state.selected_history_idx = None
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- MAIN CONTENT ----------
with main_col:
    # chat title & conversation
    chat_title = make_chat_title(st.session_state.messages)
    st.markdown(f"### {html.escape(chat_title)}")
    st.markdown("#### Conversation")
    if not st.session_state.messages:
        st.info("No messages yet. Start with a question below or load a saved chat from the left.")
    else:
        for m in st.session_state.messages:
            if m.get("role") == "user":
                st.markdown(f"**You:** {m.get('text')}")
            else:
                st.markdown(f"**Lawly:** {m.get('text')}")

    # Input form
    st.markdown("#### Ask a question")
    with st.form(key="ask_form_main"):
        user_text = st.text_input("Your question", value="", key="ask_main")
        submitted = st.form_submit_button("Enter")

    if submitted:
        q = user_text.strip()
        if q:
            st.session_state.messages.append({"role": "user", "text": q})
            try:
                gemini_client = st.session_state.get("gemini_client")
                pc = st.session_state.get("pc")
                if not gemini_client or not pc:
                    raise RuntimeError("Clients not initialized.")
                retrieved = pinecone_query(gemini_client, pc, INDEX_NAME, q, top_k=TOP_K)
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "text": f"Error retrieving sources: {e}"})
                st.rerun()

            prompt = build_rag_prompt(q, retrieved)
            try:
                answer = query_openai_chat(gemini_client, prompt)
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "text": f"Error from LLM: {e}"})
                st.rerun()

            st.session_state.messages.append({"role": "assistant", "text": answer})

            # Save history to disk for non-guest users automatically on each message (lightweight)
            if not st.session_state.get("is_guest") and st.session_state.get("user_email"):
                save_user_history(st.session_state.user_email, st.session_state.get("history", []))

            st.rerun()

# Footer legal info (sidebar)
st.sidebar.markdown("---")
st.sidebar.write("**Legal & Privacy**")
st.sidebar.caption("Lawly is for reference only. Do not input confidential data. Consult a qualified lawyer for legal decisions.")
