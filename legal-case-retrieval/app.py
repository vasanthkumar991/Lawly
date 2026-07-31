import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

from dotenv import load_dotenv
from scripts.generate_answer_llm import generate_answer
from scripts.feedback_loop import record_feedback

load_dotenv()

app = Flask(__name__)
app.secret_key = 'replace_with_your_secret_key'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, password, email):
        self.id = id
        self.username = username
        self.password = password
        self.email = email

users = [
    User(1, "alice", "123", "alice@email.com"),
    User(2, "bob", "abc", "bob@email.com")
]

@login_manager.user_loader
def load_user(user_id):
    return next((u for u in users if str(u.id) == user_id), None)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = next((u for u in users if u.username == username and u.password == password), None)
        if user:
            login_user(user)
            session['history'] = []  # Initialize chat history
            return redirect(url_for('chatbot'))
        else:
            flash('Invalid username or password!', 'danger')
    return render_template('login.html')

@app.route('/forgot', methods=['GET', 'POST'])
def forgot():
    if request.method == 'POST':
        username = request.form['username']
        user = next((u for u in users if u.username == username), None)
        if user:
            flash(f"A password reset link has been sent to {user.email} (simulated).", "info")
        else:
            flash("Username not found.", "danger")
    return render_template("forgot.html")

@app.route('/chatbot', methods=['GET', 'POST'])
@login_required
def chatbot():
    answer, sources = "", []
    history = session.get('history', [])
    if request.method == 'POST' and 'question' in request.form:
        question = request.form['question']
        dummy_chunks = [
            {"section_heading": "Section 379", "text": "Punishment for theft is imprisonment up to 3 years, or with fine, or both.", "filename": "IPC"}
        ]
        answer = generate_answer(question, dummy_chunks)
        sources = dummy_chunks

        # Add to chat history (list of dicts)
        history.append({
            "question": question,
            "answer": answer
        })
        session['history'] = history  # update session

        rating = request.form.get('rating')
        comments = request.form.get('comments')
        if rating or comments:
            record_feedback(question, answer, ", ".join([c['section_heading'] for c in sources]), rating, comments)
            flash('Thank you for your feedback!', 'success')
    return render_template('chatbot.html', name=current_user.username, answer=answer, sources=sources, history=history)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

if __name__ == "__main__":
    print("Flask Legal Chatbot running! Visit http://localhost:5000/")
    app.run(debug=True)
