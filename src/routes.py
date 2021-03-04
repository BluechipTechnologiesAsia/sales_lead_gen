from flask import Flask, send_file, send_from_directory, flash, request,\
        redirect, render_template, url_for
from app import app
from forms import PredictionForm
import numpy as np
import pandas as pd
from pickle import load
import json
from os.path import join

# loading machine learning model
model = load(open("model",'rb'))

# loading the json file 
with open("scorer.json") as f:
    data = json.load(f)

columns = ['age', 'duration', 'campaign', 'pdays', 'previous',\
           'emp_var_rate','cons_price_idx', 'cons_conf_idx',\
           'euribor3m', 'nr_employed','job', 'marital',\
           'education', 'default', 'housing', 'loan',\
           'contact', 'month', 'day_of_week', 'poutcome']

# dictionary to hold the values
dict_val = {}
for i in range(len(columns)):
    dict_val[i] = None
# assigning some constant values
dict_val[13] = 0
dict_val[7] = 1.5
dict_val[8] = 1.5
dict_val[5] = 1.5


# dataframe
dataframe = None

@app.route('/prediction')
def prediction():
    """
    """
    global dataframe
    value = model.predict(dataframe)[0]
    if value == 1:
        value = "User is more likely to buy the product"
    else:
        value = "User is less likely tp buy the product"
    return render_template('prediction.html',value = value, filename='graph.png')

@app.route('/', methods=['POST', 'GET'])
def index_page():
    """
    """
    global data,columns,dict_val, dataframe
    form = PredictionForm()
    if form.validate_on_submit():
        # creating a dataframe with the input values
        for val in form:
            if val.id in columns:
                # if the value categorical
                if val.id in data:
                    # obtaining the labeled id 
                    temp_val = data[val.id].index(val.data)
                    idx = columns.index(val.id)
                    dict_val[idx] = temp_val
                else:
                    idx = columns.index(val.id)
                    dict_val[idx] = val.data
        print(dict_val)
        arr = [val for val in dict_val.values()]
        arr = np.array([arr])
        df = pd.DataFrame(arr,columns=columns)
        dataframe = df
        print(df)
        flash(f"prediction completed!", 'success')
        return redirect(url_for('prediction'))
    return render_template('index.html', form=form)

@app.route('/show/<filename>')
def showImage(filename):
    return redirect(url_for('static',filename = join('images',filename),
                            code = 301))



if __name__ == "__main__":
    app.run(debug=True)
