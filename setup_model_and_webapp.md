# Setup venv
- `py -m venv venv`
- `venv\Scripts\activate`

# Create .flaskenv
    FLASK_ENV=development
    FLASK_APP=main.py

- `pip install python-dotenv`

# create application (already done?)
- create `main.py`
- create folder `application`
- create file `__init__.py` in application folder. contains jupyter notebook code. Copy from https://github.com/LinkedInLearning/dsm-bank-model-2870047/blob/main/application/__init__.py

# install requirements
- `pip install gunicorn`
- `pip install scikit-learn`
- `pip install pandas`

- create Procfile
    `web: gunicorn application:app`

- create requirements.txt
    `pip freeze >requirements.txt`

# create app and pipeline in heroku

# initialize bank model flask app (fix small bug in __init__.py probably due to change of version) 
- `set FLASK_APP=main.py`
- `flask run`
- on browser: `http://127.0.0.1:5000/api`



