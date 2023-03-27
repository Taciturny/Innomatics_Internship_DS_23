import os
from flask import Flask, request, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import random


alphameric = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_'

app = Flask(__name__)

basedir =os.path.abspath(os.path.dirname(__file__)) # absolute path

################SQLALCHEMY CONFIGURATION###############
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir,'database.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
################SQLALCHEMY CONFIGURATION###############


################MODEL CREATION#########################
class URL(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    original_url = db.Column(db.String(500), nullable = False)
    short_url = db.Column(db.String(20), nullable = False, unique = True)
    
    def __repr__(self):
        return f'Original URL {self.original_url}----Shorten URL {self.short_url}'

################MODEL CREATION#########################

#Create an endpoint
@app.route('/', methods =['GET', 'POST'])
def index():
    if request.method == 'POST':
        original_url= request.form['url']
        small_url = "".join(random.sample(alphameric, 4))
        new_url = URL(original_url=original_url, short_url = small_url)
        db.session.add(new_url)
        db.session.commit()

        return redirect(url_for('content'))
    else:
        return render_template('index.html')



@app.route('/<id>/<shorten>')
def redirect_url(id,shorten):
    print(id)
    url = URL.query.filter_by(id=id).first().short_url
    return redirect(url)


@app.route('/content', methods = ['GET', 'POST'])
def content():
    urls = URL.query.all()
    return render_template('contents.html', urls=urls)

if __name__ =='__main__':
    app.run(debug=True)

