from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from datetime import datetime
import DatabaseConnection
import time
from user import User

app = Flask(__name__)
app.secret_key = 'asfasfasfasqwerqwr'


# homepage direction
@app.route('/')
def index():
    return redirect('/homepage')


# homepage route
@app.route("/homepage")
def homepage():
    return render_template('homepage.html')


# login routes and methods
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.session_protection = "strong"
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(userid):
    return User.get(userid)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    if request.method == 'POST':
        if "username" in request.form and "password" in request.form:
            username = request.form['username']
            password = request.form['password']
            DatabaseConnection.database.connect_database()
            userid, userpermission = DatabaseConnection.exec_user_login(username, password)
            DatabaseConnection.database.disconnect_database()
        else:
            username = None
            userid = None
            userpermission = None
        if userid is not None:
            if userpermission == 1:
                login_user(User(id=userid, username=username))
                flash('Logged in successfully.')
                return redirect(url_for('dashboard'))
            else:
                flash('Permission denied.')
                return render_template('login.html')
        else:
            flash('Wrong username or password!')
            return render_template('login.html')


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# homepage route
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user.username)


if __name__ == '__main__':
    app.run(debug=True, host=('0.0.0.0'))
    #app.run(debug=True)
