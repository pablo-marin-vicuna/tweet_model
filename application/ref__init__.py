from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#load data
df = pd.read_csv("./car_data/car.data", header = None, names=['buying','maint','doors','persons','lug_boot','safety','class'])

X = df.iloc[:, :-1].values #all but last col
y = df.iloc[:, -1].values #last col

#onehotencoding
ohe = OneHotEncoder()
categoric_data = ohe.fit_transform(X).toarray()
categoric_df = pd.DataFrame(categoric_data)
categoric_df.columns = ohe.get_feature_names_out() #ohe.get_featureX_names()

#train model
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(categoric_df, y)

#create flask instance (this is for deployment)
app = Flask(__name__)

#create api
@app.route('/api', methods=['GET', 'POST']) # GET: can cache data but POST is more secure.
def predict():
    #get data from request
    data = request.get_json(force=True) # user input from web app
    data_categoric = np.array([data["buying"], data["maint"], data["doors"], data["persons"], data["lug_boot"], data["safety"]]) # converts input into array
    data_categoric = np.reshape(data_categoric, (1, -1)) # reformat row to column
    data_categoric = ohe.transform(data_categoric).toarray() # ohe to data, same as in training
 
    data_final = data_categoric # np.column_stack((data_age, data_balance, data_categoric))
    data_final = pd.DataFrame(data_final, dtype=object)

    #make predicon using model
    prediction = rfc.predict(data_final)
    return Response(json.dumps(prediction[0])) # convert to json back to responde object