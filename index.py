import json
import logging
import pickle

import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
app.logger.setLevel(logging.ERROR)
model = pickle.load(open('model.pkl','rb'))

data_preprocess = {}

with open('output_array.json', 'r') as json_file:
    data_preprocess = json.load(json_file)
    


@app.route('/')
def home() :
    return "Hello, World!"

@app.route('/predict',methods=['POST'])
def predict():
    df_predict = preprocess()
    prediction = model.predict(df_predict.values)
    
    labels = ['Attack','Benign','C&C','C&C-FileDownload','C&C-HeartBeat','C&C-Torii','DDoS','FileDownload','Okiru','Okiru-Attack','PartOfAHorizontalPortScan']
    
    data = {
        'id_label': str(prediction[0]),
        'label': labels[prediction[0]],
    }
    return jsonify(data)

def findReplaceNumber(field,value):
    try:
        final = data_preprocess[field][value]
        return final
    except:
        return 0 
    

def preprocess():
    
    ts = request.form['ts']
    id_resp_p = request.form['idresp_p']
    id_orig_p = request.form['idorig_p']
    orig_ip_bytes = request.form['orig_ip_bytes']
    resp_ip_bytes = request.form['resp_ip_bytes']
    history = request.form['history']
    conn_state = request.form['conn_state']

    attributes = [ts,id_resp_p,orig_ip_bytes,resp_ip_bytes, id_orig_p,history,conn_state]
    
    df = pd.DataFrame([attributes], columns=['ts', 'id.resp_p', 'orig_ip_bytes', 'resp_ip_bytes' ,'id.orig_p', 'history', 'conn_state'])
    
    # nameColumn = ['ts', 'id.resp_p', 'orig_ip_bytes', 'resp_ip_bytes' ,'id.orig_p', 'history', 'conn_state']
    # df_ditect = df[nameColumn]
    for index, row in df.iterrows():
        df.at[index, 'conn_state'] = findReplaceNumber('conn_state', row['conn_state'])
        df.at[index, 'history'] = findReplaceNumber('history',row['history'])
     
    for col in df.columns :
        df[col] = df[col].astype(float)

    return df


app.run(debug=True)


