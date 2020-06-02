import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve,plot_precision_recall_curve
from sklearn.metrics import precision_score , recall_score

def main():
    st.title("Binary Classification Web Application")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your Mushroom Edible or Poisonous")
    st.sidebar.markdown("Are your Mushroom Edible or Poisonous")
    
    
    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        
        for col in data.columns:
            data[col] = label.fit_transform(data[col])    
        return data
    
    @st.cache(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns = ['type'])
        x_train , x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=0)
        return x_train , x_test,y_train,y_test
    
    def plot_metrics(metrics_list):
        if 'confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model, x_test,y_test, display_labels=class_names)
            st.pyplot()
        
        if 'confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model, x_test,y_test, display_labels=class_names)
            st.pyplot()
        
        if 'confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model, x_test,y_test, display_labels=class_names)
            st.pyplot()
    
    df = load_data()
    x_train , x_test,y_train,y_test = split(df)
    class_names = ['edible','poisonous']
    st.sidebar.subheader("Choose Classfier")
    classfier = st.sidebar.selectbox("Classfier","Support Vector Machine (SVM),Logistic Regression,Random Forest Classifer")
    
    
    
    
    if st.sidebar.checkbox("Show raw Data", False):
        st.subheader("Mushroom Data set Classification")
        st.write(df)
        

if __name__ == '__main__':
    main()