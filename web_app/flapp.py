from flask import Flask, render_template, request, redirect, url_for, session
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # for user sessions

model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

users = {'admin': 'password'}  # Replace with real db or file

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('predict'))
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        features = [float(request.form[x]) for x in ['ph', 'hardness', 'solids', 'chloramines',
                                                      'sulfate', 'conductivity', 'organic_carbon',
                                                      'trihalomethanes', 'turbidity']]
        scaled = scaler.transform([features])
        pred = model.predict(scaled)[0]
        return render_template('result.html', prediction=pred)
    return render_template('predict.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
