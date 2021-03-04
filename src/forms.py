from flask_wtf import FlaskForm
from wtforms import DecimalField ,IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, Length, Regexp
import json

# loading the json file
with open('scorer.json') as f:
    data = json.load(f)


class PredictionForm(FlaskForm):
    """
    """
    age = IntegerField(label = 'age',validators = [DataRequired()])
    job = SelectField(label='Job', choices=data['job'], validators = [DataRequired()])
    marital = SelectField(label='Marital Status', choices=data['marital'], validators = [DataRequired()])
    education = SelectField(label="Education", choices=data['education'], validators = [DataRequired()])
#    default = SelectField(label="default",choices=data['default'], validators = [DataRequired()])
    housing = SelectField(label="housing",choices=data['housing'], validators = [DataRequired()])
    loan = SelectField(label="loan",choices=data['loan'], validators = [DataRequired()])
    contact = SelectField(label="contact",choices=data['contact'], validators = [DataRequired()])
    month = SelectField(label="month",choices=data['month'], validators = [DataRequired()])
    day_of_week = SelectField(label="day",choices=data['day_of_week'], validators = [DataRequired()])
    duration = IntegerField(label="call duration", validators = [DataRequired()])
    campaign = IntegerField(label="Number of contacts performed during this\
                            campaign and for this client", validators = [DataRequired()])
    pdays = IntegerField(label="Numner of Days that passed by after the client\
                         was last contacted from previous campaign", validators = [DataRequired()])
    previous = IntegerField(label="Number of contacts performed before this\
                            campaign and for this client", validators = [DataRequired()])
    poutcome = SelectField(label="Outcome of previous\
                           campaign",choices=data['poutcome'], validators = [DataRequired()])
#    emp_var_rate = IntegerField(label="Employment Variation Rate", validators = [DataRequired()])
    cons_price_idx = IntegerField(label="Consumer Price Index", validators = [DataRequired()])
#    cons_conf_idx = IntegerField(label="Consumer Confidence Index", validators = [DataRequired()])
#    euribor3m = IntegerField(label="euribor 3 month rate", validators = [DataRequired()])
    nr_employed = IntegerField(label="Number of Employees", validators = [DataRequired()])
    submit = SubmitField('Predict')
