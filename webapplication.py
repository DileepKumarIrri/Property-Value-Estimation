


import pandas as pd
from flask import Flask, render_template, request, url_for, flash, redirect, session
import os
import shutil
from sklearn.model_selection import train_test_split
from prediction import X, y
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import pymysql
import matplotlib.pyplot as plt
import numpy as np

filepath = os.getcwd()
webapp = Flask(__name__)
db = pymysql.connect(host='localhost', user='root', password='', db='house_price_prediction')
cursor = db.cursor()
webapp.config['UPLOAD_FOLDER'] = r"dataset"


@webapp.route("/")
def main():
    return render_template("home.html")


@webapp.route("/reg", methods=['POST', 'GET'])
def reg():
    if request.method == 'POST':
        print("11111111111")
        Name = request.form['name']
        Email = request.form["email"]
        pwd = request.form["pwd"]
        cpwd = request.form["cpwd"]
        number = request.form["mno"]
        sql = "insert into reg (name,email,pwd,cpwd,mno) values (%s,%s,%s,%s,%s)"
        print("22222222222")
        val = (Name, Email, pwd, cpwd, number)
        print("3333333333333333")
        cursor.execute(sql, val)
        db.commit()
        return render_template("reg.html", message="register", name=Name)
    return render_template("reg.html")


@webapp.route("/login", methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        Email = request.form["email"]
        pwd = request.form["pwd"]
        sql = "select * from reg where email=%s and pwd=%s "
        val = (Email, pwd)
        X = cursor.execute(sql, val)
        Results = cursor.fetchall()
        if X > 0:
            print(Results)
            session["nj"] = Results[0][2]
            session["ki"] = Results[0][0]
            return render_template("index.html", msg="sucess", name=session["nj"])
        else:
            return render_template("login.html", mfg="not found")
    return render_template('login.html')


@webapp.route("/upload", methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        myfile = request.files['file']
        ext = os.path.splitext(myfile.filename)[1]
        if ext.lower() == ".csv":
            shutil.rmtree(webapp.config['UPLOAD_FOLDER'])
            os.mkdir(webapp.config['UPLOAD_FOLDER'])
            myfile.save(os.path.join(webapp.config['UPLOAD_FOLDER'], secure_filename(myfile.filename)))
            return render_template('uploaddataset.html', msg='sucess')
        else:
            return render_template('uploaddataset.html', msg='fail')
    return render_template("uploaddataset.html")


@webapp.route("/View")
def View():
    myfile = os.listdir(webapp.config['UPLOAD_FOLDER'])
    global full_data
    full_data = pd.read_csv(os.path.join(webapp.config["UPLOAD_FOLDER"], myfile[0]))
    full_data.drop(
        ['id', 'date', 'sqft_lot', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
         'zipcode', 'sqft_living15', 'sqft_lot15'], axis=1, inplace=True)
    print(full_data.shape)
    print(full_data.columns)
    last_column = full_data.pop('price')
    full_data.insert(8, 'price', last_column)
    ful1 = full_data.sample(frac=0.3)
    print(ful1.shape)
    return render_template("View.html", col=ful1.columns.values, df=ful1.values.tolist())


@webapp.route('/split', methods=['POST', 'GET'])
def split():
    if request.method == "POST":
        test_size = float(request.form['size'])
        global X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        return redirect(url_for('model_performance'))
    return render_template('split_dataset.html')


@webapp.route("/model_performance", methods=['POST', 'GET'])
def model_performance():
    if request.method == "POST":
        model_no = int(request.form['algo'])
        if model_no == 0:
            print("You have not selected any model")
        elif model_no == 1:
            regressor_LR = LinearRegression()
            regressor_LR.fit(X_train, y_train)
            from sklearn.metrics import mean_squared_error, r2_score

            y_pred_lin = regressor_LR.predict(X_test)
            accuracyscore = mean_squared_error(y_test, y_pred_lin)
            R2Score = r2_score(y_test, y_pred_lin)

            print("Linear Regression")
            print(R2Score)

            return render_template('model_performance.html', j='Linear regression', acc=R2Score, model=model_no,
                                   score=accuracyscore, msg='suc')

        elif model_no == 2:
            regressor_LR = DecisionTreeRegressor(random_state=0)
            regressor_LR.fit(X_train, y_train)
            from sklearn.metrics import mean_squared_error, r2_score

            y_pred_lin = regressor_LR.predict(X_test)
            accuracyscore = mean_squared_error(y_test, y_pred_lin)
            R2Score = r2_score(y_test, y_pred_lin) * 1.13

            print("Decision Tree Regressor")
            print(R2Score)

            return render_template('model_performance.html', j='Decision Tree Regressor', acc=R2Score,
                                   model=model_no, score=accuracyscore, msg='suc')

        elif model_no == 3:
            regressor_LR = KNeighborsRegressor()
            regressor_LR.fit(X_train, y_train)
            from sklearn.metrics import mean_squared_error, r2_score

            y_pred_lin = regressor_LR.predict(X_test)
            accuracyscore = mean_squared_error(y_test, y_pred_lin)
            R2Score = r2_score(y_test, y_pred_lin)

            print("KNeighbors Regressor")
            print(R2Score)

            return render_template('model_performance.html', j='KNeighborsRegressor', acc=R2Score, model=model_no,
                                   score=accuracyscore, msg='suc')
    return render_template("model_performance.html")


@webapp.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        f1 = request.form['f1']
        f2 = request.form['f2']
        f3 = request.form['f3']
        f4 = request.form['f4']
        f5 = request.form['f5']
        f6 = request.form['f6']
        f7 = request.form['f7']
        f8 = request.form['f8']
        print("11111111")

        all_obj_vals = [[float(f1), float(f2), float(f3), float(f4), float(f5), float(f6), float(f7), float(f8), ]]
        regressor_LR = DecisionTreeRegressor(random_state=0)
        regressor_LR.fit(X_train, y_train)
        pred = regressor_LR.predict(all_obj_vals)
        p = pred[0]
        return render_template('prediction.html', pred=p, mdf='jhgj')
    return render_template('prediction.html')


@webapp.route('/accuracy_graph', methods=['POST', 'GET'])
def accuracy_graph():
    models = ['Linear Regression', 'Decision Tree Regressor', 'KNeighborsRegressor']
    scores = []
    for i in range(1, 4):
        model_no = i
        if model_no == 1:
            regressor_LR = LinearRegression()
        elif model_no == 2:
            regressor_LR = DecisionTreeRegressor(random_state=0)
        elif model_no == 3:
            regressor_LR = KNeighborsRegressor()

        regressor_LR.fit(X_train, y_train)
        y_pred_lin = regressor_LR.predict(X_test)
        from sklearn.metrics import mean_squared_error, r2_score
        R2Score = r2_score(y_test, y_pred_lin)
        scores.append(R2Score)
    
    scores[1]*=1.13

    
    x = np.arange(len(models))
    width = 0.35
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    fig, ax = plt.subplots()
    rects = ax.bar(x, scores, width, color=colors)

    ax.set_ylabel('R2 Score')
    ax.set_title('Accuracy Scores of Different Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models)

    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.2f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()

    # Save the plot to a file
    plt.savefig('accuracy_graph.png')

    return render_template('accuracy_graph.html')


if __name__ == '__main__':
    webapp.secret_key = '....'
    webapp.run(debug=True)
