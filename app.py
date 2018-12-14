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


# analyze route
@app.route("/analyze")
@login_required
def analyze():
    pie_data = ["141", "38", "6", "330", "70", "14", "720"]
    pie_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']
    mooddata = [76.7, 73.8, 88.6, 70.0, 78.2, 73.3, 76.4]
    return render_template('analyze.html', user=current_user.username, pie_data=pie_data, pie_labels=pie_labels, mooddata= mooddata)

# statistics route
@app.route("/statistics")
@login_required
def statistics():
    daily_record = [
        ["2018/11/27", 3, 76.4],
        ["2018/11/26", 5, 73.3],
        ["2018/11/25", 4, 78.2],
        ["2018/11/24", 7, 70.0],
        ["2018/11/23", 2, 88.6],
        ["2018/11/22", 6, 73.8],
        ["2018/11/21", 4, 76.7]
    ]
    hourly_record = [
        ["2018/11/27", "19:00", 74.1],
        ["2018/11/27", "18:00", 78.3],
        ["2018/11/27", "17:00", 76.8],
        ["2018/11/26", "22:00", 78.3],
        ["2018/11/26", "21:00", 70.2],
        ["2018/11/26", "20:00", 69.4],
        ["2018/11/26", "19:00", 67.5],
        ["2018/11/26", "18:00", 81.1],
        ["2018/11/25", "21:00", 77.5],
        ["2018/11/25", "20:00", 81.1],
        ["2018/11/25", "19:00", 67.5],
        ["2018/11/25", "18:00", 81.1],
    ]
    return render_template('statistics.html', user=current_user.username, daily_record=daily_record, hourly_record=hourly_record)



if __name__ == '__main__':
    app.run(debug=True, host=('0.0.0.0'))
    #app.run(debug=True)
