from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from argparse import ArgumentParser
#from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace,Dataset
from azureml.data.dataset_factory import TabularDatasetFactory


#dataset=pd.read_csv('heart_failure_clinical_records_dataset.csv')
path_url="https://raw.githubusercontent.com/Ankita03-dell/AZMLND_Capstone_Trial1/main/heart_failure_clinical_records_dataset.csv"
ds=TabularDatasetFactory.from_delimited_files(path=path_url)
x = ds[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]
y = ds[['DEATH_EVENT']]

#Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state=0)  
        
run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()
    
    #ds = pd.read_csv('./heart_failure_clinical_records_dataset.csv')
    #create Tabular Dataset using TabularDatasetFactory
    #path_url="https://raw.githubusercontent.com/Ankita03-dell/AZMLND_Capstone_Trial1/main/heart_failure_clinical_records_dataset.csv"
    #ds=TabularDatasetFactory.from_delimited_files(path=path_url)
    #dataset=pd.read_csv('heart_failure_clinical_records_dataset.csv')
    #x = dataset[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]
    #y = dataset[['DEATH_EVENT']]
    #x= ds.drop('DEATH_EVENT', axis=1)
    #y= ds['DEATH_EVENT']
 
    #Split data into train and test sets
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state=0)  
        
    #run = Run.get_context()
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename= 'outputs/model.pk1')
    

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
