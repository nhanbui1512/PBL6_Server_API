import json
import pickle

import pandas as pd

model = pickle.load(open('model.pkl','rb'))
data_preprocess = {}

with open('output_array.json', 'r') as json_file:
    data_preprocess = json.load(json_file)
    
def findReplaceNumber(field,value):
    try:
        final = data_preprocess[field][value]
        return final
    except:
        return 0 
    

    
df = pd.read_csv('./data_test.csv', usecols=['ts', 'id.resp_p', 'orig_ip_bytes', 'resp_ip_bytes' ,'id.orig_p', 'history', 'conn_state','detailed-label'])
X_train = df[['ts', 'id.resp_p', 'orig_ip_bytes', 'resp_ip_bytes' ,'id.orig_p', 'history', 'conn_state']]
for index, row in df.iterrows():
    X_train.at[index, 'conn_state'] = findReplaceNumber('conn_state', row['conn_state'])
    X_train.at[index, 'history'] = findReplaceNumber('history',row['history'])

for col in X_train.columns :
    X_train[col] = X_train[col].astype(float)


prediction = model.predict(X_train.values)

df_label = pd.read_csv('data_test.csv', usecols=['detailed-label'])
df_label['detailed-label'] = df_label['detailed-label'].replace('Attack',0)
# df_label['detailed-label'] = df_label['detailed-label'].replace('Benign',1)
df_label['detailed-label'] = df_label['detailed-label'].replace('-',1)
df_label['detailed-label'] = df_label['detailed-label'].replace('C&C',2)
df_label['detailed-label'] = df_label['detailed-label'].replace('C&C-FileDownload',3)
df_label['detailed-label'] = df_label['detailed-label'].replace('C&C-HeartBeat',4)
df_label['detailed-label'] = df_label['detailed-label'].replace('C&C-Torii',5)
df_label['detailed-label'] = df_label['detailed-label'].replace('DDoS',6)
df_label['detailed-label'] = df_label['detailed-label'].replace('FileDownload',7)
df_label['detailed-label'] = df_label['detailed-label'].replace('Okiru',8)
df_label['detailed-label'] = df_label['detailed-label'].replace('Okiru-Attack',9)
df_label['detailed-label'] = df_label['detailed-label'].replace('PartOfAHorizontalPortScan',10)

count = 0 
for i in range(0, len(prediction) -1 ):
    if(prediction[i] == df_label['detailed-label'][i]):
        count = count + 1

print(count/len(prediction)*100)



from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

def matrix_confusion(y_true, y_pred):
    labels_true = unique_labels(y_true)
    labels_pred = unique_labels(y_pred)
    labels = np.union1d(labels_true, labels_pred)
    column = [f'Predict {label}' for label in labels]
    indices = [f'Actual {label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(y_true, y_pred),
                         columns=column, index=indices)
    return table

def sum_array(arr):
    total = 0
    for num in arr:
        total += num
    return total

def calculate_precision_recall(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    TP_total = 0
    TP_FP_total = 0
    TP_FN_total = 0
    precision_averaging = []
    recall_averaging = []
    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        # tong TP, TP_FP, TP_FN
        TP_total += TP
        TP_FP_total += (TP + FP)
        TP_FN_total += (TP + FN)
        # ty le tung star
        precision[i] = TP / (TP + FP)
        recall[i] = TP / (TP + FN)
    # trung bình vi mô
    precision_Micro_averaging = TP_total / TP_FP_total
    precision_averaging.append(precision_Micro_averaging)
    recall_Micro_averaging = TP_total / TP_FN_total
    recall_averaging.append(recall_Micro_averaging)
    # trung bình vĩ mô
    precision_Macro_averaging = sum_array(precision) / num_classes
    precision_averaging.append(precision_Macro_averaging)
    recall_Macro_averaging = sum_array(recall) / num_classes
    recall_averaging.append(recall_Macro_averaging)

    precision = np.nan_to_num(precision, nan=0)
    recall = np.nan_to_num(recall, nan=0)

    return np.mean(precision), np.mean(recall), precision_averaging, recall_averaging

def show_matrix_confusion(Y_test, y_pred_RandomForest):

    CM = matrix_confusion(Y_test, y_pred_RandomForest)
    CM_precision_recall = CM.copy()
    precision, recall, precision_averaging, recall_averaging = calculate_precision_recall(CM_precision_recall.values)
    
    # trung bình vi mô và vĩ mô
    column = ['Micro_averaging', 'Macro_averaging']
    row = ['precision', 'recall']
    table_average = pd.DataFrame([precision_averaging, recall_averaging], columns=column, index= row)
    
    print("Ma tran nham lan \n", CM_precision_recall.values)
    sns.heatmap(CM, annot=True, fmt='d', cmap='viridis')
    print("\nBảng đánh giá hiệu suất trung bình vi mô và vĩ mô của mô hình \n", table_average)

    return precision, recall

precision_rf, recall_rf = show_matrix_confusion(df_label['detailed-label'], prediction)
