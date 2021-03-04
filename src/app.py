from flask import Flask
from os.path import abspath, join
from os import getcwd


app = Flask(__name__)
app.secret_key = "secret key"
