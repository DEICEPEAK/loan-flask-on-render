import os
from flask import Flask, render_template, session, url_for, redirect, flash
from wtforms import StringField, SubmitField, BooleanField, SelectField
from wtforms.validators import DataRequired, NumberRange, ValidationError, InputRequired
import numpy as np
import pandas as pd
from flask_wtf import FlaskForm
import joblib

def return_prediction(model, scaler, col_name, sample_json):
    age = sample_json["Age"]
    income = sample_json["Income"]
    l_amount = sample_json["LoanAmount"]
    cd_score = sample_json["CreditScore"]
    month_emp = sample_json["MonthsEmployed"]
    num_cl = sample_json["NumCreditLines"]
    int_rate = sample_json["InterestRate"]
    loan_term = sample_json["LoanTerm"]
    dti_ratio = sample_json["DTIRatio"]
    edu = sample_json["Education"]
    emp_type = sample_json["EmploymentType"]
    marital_s = sample_json["MaritalStatus"]
    has_mort = sample_json["HasMortgage"]
    has_depd = sample_json["HasDependents"]
    loan_purp = sample_json["LoanPurpose"]
    has_cosg = sample_json["HasCoSigner"]

    cat_df = pd.DataFrame([[edu, emp_type, marital_s, has_mort, has_depd, loan_purp, has_cosg]],
                          columns=['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'])
    cat_encoded = pd.get_dummies(cat_df, drop_first=True)

    num_df = pd.DataFrame([[age, income, l_amount, cd_score, month_emp, num_cl, int_rate, loan_term, dti_ratio]],
                          columns=['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                                   'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'])
    combined_df = pd.concat([num_df, cat_encoded], axis=1)
    combined_df = combined_df.reindex(columns=col_name, fill_value=0)

    loan = combined_df.values
    num_scaled = scaler.transform(loan[:, :9])
    loan = np.hstack([num_scaled, loan[:, 9:]])

    prediction_proba = model.predict_proba(loan)[0]
    default_proba = prediction_proba[1]

    return f'There is a {default_proba * 100:.2f}% probability that this applicant will default.'

app = Flask(__name__)
app.config["SECRET_KEY"] = 'mysecretkey'

class LoanForm(FlaskForm):
    def float_validator(form, field):
        try:
            field.data = float(field.data)
        except ValueError:
            raise ValidationError('Invalid input: Must be a number')

    ag = StringField("AGE", validators=[DataRequired(), float_validator, NumberRange(min=18, max=100)], render_kw={"placeholder": "e.g., 30"})
    inc = StringField("INCOME", validators=[DataRequired(), float_validator, NumberRange(min=0)], render_kw={"placeholder": "e.g., 50000"})
    l_amt = StringField("LOAN AMOUNT", validators=[DataRequired(), float_validator, NumberRange(min=0)], render_kw={"placeholder": "e.g., 100000"})
    c_sco = StringField("CREDIT SCORE", validators=[DataRequired(), float_validator, NumberRange(min=300, max=850)], render_kw={"placeholder": "e.g., 700"})
    m_emp = StringField("MONTHS EMPLOYED", validators=[DataRequired(), float_validator, NumberRange(min=0)], render_kw={"placeholder": "e.g., 24"})
    n_cl = StringField("NUMBER OF CREDIT LINES", validators=[DataRequired(), float_validator, NumberRange(min=0)], render_kw={"placeholder": "e.g., 3"})
    i_rate = StringField("INTEREST RATE", validators=[DataRequired(), float_validator, NumberRange(min=0.0, max=100.0)], render_kw={"placeholder": "e.g., 5.5"})
    l_term = StringField("LOAN TERM", validators=[DataRequired(), float_validator, NumberRange(min=1)], render_kw={"placeholder": "e.g., 36"})
    dti = StringField("DTI RATIO", validators=[DataRequired(), float_validator, NumberRange(min=0.0, max=100.0)], render_kw={"placeholder": "e.g., 15.6"})
    mortg = BooleanField("HAS MORTGAGE?")
    deped = BooleanField("HAS DEPENDENTS?")
    cosign = BooleanField("HAS COSIGNER?")
    ed = SelectField("EDUCATION", choices=[("Bachelor's", "Bachelor's"), ("Phd", "Phd"), ("Master's", "Master's"), ("High School", "High School")], validators=[DataRequired()])
    em_typ = SelectField("EMPLOYMENT TYPE", choices=[("Full-time", "Full-time"), ("Unemployed", "Unemployed"), ("Self-employed", "Self-employed"), ("Part-time", "Part-time")], validators=[DataRequired()])
    mar_st = SelectField("MARITAL STATUS", choices=[("Married", "Married"), ("Divorced", "Divorced"), ("Single", "Single")], validators=[DataRequired()])
    l_purp = SelectField("LOAN PURPOSE", choices=[("Auto", "Auto"), ("Business", "Business"), ("Education", "Education"), ("Home", "Home"), ("Other", "Other")], validators=[DataRequired()])

    submit = SubmitField("Apply")

@app.route("/", methods=['GET', 'POST'])
def index():
    form = LoanForm()

    if form.validate_on_submit():
        session['ag'] = form.ag.data
        session['inc'] = form.inc.data
        session['l_amt'] = form.l_amt.data
        session['c_sco'] = form.c_sco.data
        session['m_emp'] = form.m_emp.data
        session['n_cl'] = form.n_cl.data
        session['i_rate'] = form.i_rate.data
        session['l_term'] = form.l_term.data
        session['dti'] = form.dti.data
        session['ed'] = form.ed.data
        session['em_typ'] = form.em_typ.data
        session['mar_st'] = form.mar_st.data
        session['mortg'] = form.mortg.data
        session['deped'] = form.deped.data
        session['l_purp'] = form.l_purp.data
        session['cosign'] = form.cosign.data

        log_data = {
            'Age': form.ag.data,
            'Income': form.inc.data,
            'LoanAmount': form.l_amt.data,
            'CreditScore': form.c_sco.data,
            'MonthsEmployed': form.m_emp.data,
            'NumCreditLines': form.n_cl.data,
            'InterestRate': form.i_rate.data,
            'LoanTerm': form.l_term.data,
            'DTIRatio': form.dti.data,
            'Education': form.ed.data,
            'EmploymentType': form.em_typ.data,
            'MaritalStatus': form.mar_st.data,
            'HasMortgage': form.mortg.data,
            'HasDependents': form.deped.data,
            'LoanPurpose': form.l_purp.data,
            'HasCoSigner': form.cosign.data
        }

        # Log data to CSV
        log_df = pd.DataFrame([log_data])
        if not os.path.isfile('log.csv'):
            log_df.to_csv('log.csv', index=False)
        else:
            log_df.to_csv('log.csv', mode='a', header=False, index=False)

        return redirect(url_for('loan_prediction'))
    return render_template('home.html', form=form)

lr_model = joblib.load('final_lr_model.pkl')
lr_scaler = joblib.load("lr_scaler.pkl")
col_name = joblib.load("col_name.pkl")

@app.route('/loan_prediction')
def loan_prediction():
    try:
        content = {
            'Age': float(session['ag']),
            'Income': float(session['inc']),
            'LoanAmount': float(session['l_amt']),
            'CreditScore': float(session['c_sco']),
            'MonthsEmployed': float(session['m_emp']),
            'NumCreditLines': float(session['n_cl']),
            'InterestRate': float(session['i_rate']),
            'LoanTerm': float(session['l_term']),
            'DTIRatio': float(session['dti']),
            'Education': session['ed'],
            'EmploymentType': session['em_typ'],
            'MaritalStatus': session['mar_st'],
            'HasMortgage': session['mortg'],
            'HasDependents': session['deped'],
            'LoanPurpose': session['l_purp'],
            'HasCoSigner': session['cosign']
        }

        results = return_prediction(lr_model, lr_scaler, col_name, content)

        # Log results to CSV
        log_df = pd.read_csv('log.csv')
        log_df.at[log_df.index[-1], 'Prediction'] = results
        log_df.to_csv('log.csv', index=False)

    except Exception as e:
        results = str(e)

    return render_template('loan_prediction.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
