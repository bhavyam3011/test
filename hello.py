import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
st. set_page_config(layout="wide")
# chart_data = pd.DataFrame(
#      np.random.randn(20, 3),
#      columns=['a', 'b', 'c'])

# st.line_chart(chart_data)
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)

y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)





tab1,tab2,tab3 = st.tabs(['ABOUT DATASET','MANIPULATIONS', 'TESTING'])

tab1.subheader('Dataset content')
tab1.write('''The dataset contains transactions made by credit cards in September 2013 by European cardholders. \n
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. 

V1 to V28: Principal components from PCA.

Time: Seconds elapsed between each transaction and the first transaction.

Amount: Transaction amount, useful for cost-sensitive learning.

Class: Indicates fraud (1) or no fraud (0).
''')

legit = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]

tab1.subheader("Legit Transactions")
tab1.dataframe(legit)
tab1.write("Legit Transactions Numerical values")
tab1.dataframe(legit.describe())
tab1.subheader("Fraudulent Transactions")
tab1.dataframe(fraud)
tab1.write("Fraudulent Transactions Numerical values")
tab1.dataframe(fraud.describe())

tab3.write(accuracy)
inputabc = tab3.text_input('Enter comma seperated texting values')
splittedi = inputabc.split(',')
submit = tab3.button("submit")
if submit: 
    features =  np.asarray(splittedi,dtype = np.float64)
    prediction  = model.predict(features.reshape(1,-1))

    if(prediction[0]==0):
        tab3.subheader("Legitimate")
    else:
        tab3.subheader("Fraud")

tab3.write('\ncopy this to test')
tab3.write('35770,-0.803507779942207,1.2997628733188,1.10400267222195,1.49206436943163,-0.436539336184394,-0.553334521499083,0.775195746479099,0.228989223581682,-1.22337112060685,-0.369636028950297,-0.559217024961813,-0.271404177238908,-0.224341929274163,0.869954709697028,1.56746870888431,-0.680423833162871,0.489363102235665,0.178621048037021,1.55026030984739,0.199603857056867,0.118395656440999,0.122652912947576,-0.112476864203786,0.381002226351728,0.471502763378559,-0.0723280466590595,-0.0640083624512651,0.0242035606518561,84.59')
tab3.write('copy this to test2')
tab3.write('43494,-1.27813773524814,0.716241580332591,-1.14327922918382,0.217804800631468,-1.29389046969973,-1.16895152511977,-2.56418198278833,0.204532467641239,-1.61115462297744,-1.25028553742691,3.36736102310858,-4.58309584545642,-0.806939683730076,-5.20830522179452,0.525823682164556,-1.81591431259746,-5.24911146794949,-2.19669120679566,2.10905264474381,0.817203155491914,0.490182667533974,0.470427403079725,-0.126261344339209,-0.126644458784455,-0.661907559467122,-0.349792996571914,0.454851068382725,0.137843477172283,24.9')
