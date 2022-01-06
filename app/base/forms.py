# -*- encoding: utf-8 -*-

from flask_wtf import FlaskForm
from wtforms import TextField, PasswordField, DateField, SelectField, FileField
from wtforms.validators import InputRequired, Email, DataRequired

## login and registration

class LoginForm(FlaskForm):
    username = TextField     ('Username', id='username_login'   , validators=[DataRequired()])
    password = PasswordField ('Password', id='pwd_login'        , validators=[DataRequired()])

class CreateAccountForm(FlaskForm):
    username  = TextField('Username'     , id='username_create' , validators=[DataRequired()])
    email     = TextField('Email'        , id='email_create'    , validators=[DataRequired(), Email()])
    password  = PasswordField('Password' , id='pwd_create'      , validators=[DataRequired()])
    firstname = TextField('First Name'   , id='firstname'       , validators=[DataRequired()])
    lastname  = TextField('Last Name'    , id='lastname'        , validators=[DataRequired()])
    age       = TextField('Age'          , id='age'             , validators=[DataRequired()])
    phone     = TextField('Contact No.'  , id='phone'           , validators=[DataRequired()])
    level     = SelectField('Level'      , id='level'           , choices=["Under-Graduate","Graduate","Post-Graduate"], validators=[DataRequired()])
    gender    = SelectField('Gender'     , id='gender'          , choices=["Male","Female","Other"] , validators=[DataRequired()])
    photo     = FileField('Photo'        , id='photo'           )