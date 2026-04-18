from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.svm import SVC

global filename
global df, X_train, X_test, y_train, y_test

def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Fill missing values with mode for each column
    df.fillna(df.mode().iloc[0], inplace=True)
    
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(len(df)) + "\n")

def splitdataset(): 
    global df, X_train, X_test, y_train, y_test

    # Encode string columns to numerical values
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])

    X = df[["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]]
    
    # Fill NaN values with mode for each column in X
    X.fillna(X.mode().iloc[0], inplace=True)

    # Categorize AQI into discrete classes
    bins = [0, 50, 100, 150, 200, 300, 400, 500]
    labels = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous', 'Severely Hazardous']
    df['AQI_Category'] = pd.cut(df['AQI'], bins=bins, labels=labels)
    
    y = df['AQI_Category']
    
    # Fill NaN values in y with mode
    y.fillna(y.mode()[0], inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    

    # Display dataset split information
    text.delete('1.0', END)
    text.insert(END, "Dataset split\n")
    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(X_train)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(X_test)) + "\n")
    
    # Display shapes of X_train, X_test, y_train, y_test
    text.insert(END, "\nShape of X_train: " + str(X_train.shape) + "\n")
    text.insert(END, "Shape of X_test: " + str(X_test.shape) + "\n")
    text.insert(END, "Shape of y_train: " + str(y_train.shape) + "\n")
    text.insert(END, "Shape of y_test: " + str(y_test.shape) + "\n\n")


def LR():
    global lr_acc, lr_precision, lr_recall, lr_f1
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, y_pred)
    lr_precision = precision_score(y_test, y_pred, average='weighted')
    lr_recall = recall_score(y_test, y_pred, average='weighted')
    lr_f1 = f1_score(y_test, y_pred, average='weighted')
    text.delete('1.0', END)

    result_text = f'Accuracy for Logistic Regression is {lr_acc * 100}%\n'
    result_text += f'Precision for Logistic Regression is {lr_precision}\n'
    result_text += f'Recall for Logistic Regression is {lr_recall}\n'
    result_text += f'F1 Score for Logistic Regression is {lr_f1}\n'
    text.insert(END, result_text)

def random_forest():
    global rf_acc, rf_precision, rf_recall, rf_f1,rf
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred)
    rf_precision = precision_score(y_test, y_pred, average='weighted')
    rf_recall = recall_score(y_test, y_pred, average='weighted')
    rf_f1 = f1_score(y_test, y_pred, average='weighted')
    text.delete('1.0', END)

    result_text = f'Accuracy for Random Forest is {rf_acc * 100}%\n'
    result_text += f'Precision for Random Forest is {rf_precision}\n'
    result_text += f'Recall for Random Forest is {rf_recall}\n'
    result_text += f'F1 Score for Random Forest is {rf_f1}\n'
    text.insert(END, result_text)


def Run_DT():
    global dt_acc, dt_precision, dt_recall, dt_f1
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, y_pred)
    dt_precision = precision_score(y_test, y_pred, average='weighted')
    dt_recall = recall_score(y_test, y_pred, average='weighted')
    dt_f1 = f1_score(y_test, y_pred, average='weighted')
    text.delete('1.0', END)

    result_text = f'Accuracy for Decision Tree is {dt_acc * 100}%\n'
    result_text += f'Precision for Decision Tree is {dt_precision}\n'
    result_text += f'Recall for Decision Tree is {dt_recall}\n'
    result_text += f'F1 Score for Decision Tree is {dt_f1}\n'
    text.insert(END, result_text)


def Run_SVM():
    global svm_acc, svm_precision, svm_recall, svm_f1
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred)
    svm_precision = precision_score(y_test, y_pred, average='weighted')
    svm_recall = recall_score(y_test, y_pred, average='weighted')
    svm_f1 = f1_score(y_test, y_pred, average='weighted')
    text.delete('1.0', END)

    result_text = f'Accuracy for SVM is {svm_acc * 100}%\n'
    result_text += f'Precision for SVM is {svm_precision}\n'
    result_text += f'Recall for SVM is {svm_recall}\n'
    result_text += f'F1 Score for SVM is {svm_f1}\n'
    text.insert(END, result_text)


import matplotlib.pyplot as plt

def pie_chart():
  # Ensure 'df' is accessible globally
  global df

  # Get the counts for each AQI category
  category_counts = df['AQI_Category'].value_counts().sort_values(ascending=False)

  # Extract category labels and counts for the pie chart
  category_labels = category_counts.index.to_numpy()
  category_values = category_counts.to_numpy()

  # Create a color list for the pie chart slices
  colors = ['skyblue', 'lightgreen', 'yellow', 'orange', 'red', 'purple', 'violet']  # Adjust colors as desired

  # Create the pie chart with customization
  plt.figure(figsize=(8, 8))  # Set the figure size
  plt.pie(category_values, labels=category_labels, autopct="%1.1f%%", startangle=140, colors=colors)
  plt.title("Air Quality Index Category Distribution", fontsize=16)
  plt.axis('equal')  # Equal aspect ratio ensures a circular pie chart

  # Add a legend
  plt.legend(category_labels, loc="upper left", bbox_to_anchor=(1, 1))

  plt.show()

def plot_results():
    """
    Generates a bar chart to visualize the accuracy of each algorithm.
    """
    # Ensure evaluation metrics are available
    if not all([lr_acc, rf_acc, dt_acc, svm_acc]):
        messagebox.showerror("Error", "Please run the Machine Learning models first to generate evaluation metrics.")
        return

    # Algorithm names and their corresponding accuracies
    algorithms = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'SVM']
    accuracies = [lr_acc, rf_acc, dt_acc, svm_acc]

    # Create the bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(algorithms, accuracies, color=['skyblue', 'lightgreen', 'yellow', 'orange'])
    plt.xlabel('Machine Learning Algorithm')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison of Machine Learning Models')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()

    plt.show()


def predict():
    global rf  # Ensure 'rf' is accessible globally

    # Open file manager to select CSV file
    filename = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

    if filename:
        # Read the selected CSV file
        input_data = pd.read_csv(filename)

        # Fill missing values with mode for each column
        input_data.fillna(input_data.mode().iloc[0], inplace=True)

        # Preprocess input data (if needed)
        label_encoder = LabelEncoder()
        for column in input_data.columns:
            if input_data[column].dtype == 'object':
                input_data[column] = label_encoder.fit_transform(input_data[column])

        # Extract features for prediction
        X_pred = input_data[["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]]
        
        # Fill NaN values in X_pred with mode
        X_pred.fillna(X_pred.mode().iloc[0], inplace=True)

        # Perform prediction using RandomForestClassifier
        y_pred = rf.predict(X_pred)
        print(y_pred)
        text.insert(END,y_pred)




main = tk.Tk()
main.title("AIR QUALITY PREDICTION USING MACHINE LEARNING") 
main.geometry("1600x1500")

font = ('times', 16, 'bold')
title = tk.Label(main, text='AIR QUALITY PREDICTION USING MACHINE LEARNING',font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)           
title.config(height=3, width=145)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = tk.Text(main, height=20, width=180)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = tk.Button(main, text="Upload Dataset", command=upload, bg="sky blue", width=15)
uploadButton.place(x=50, y=600)
uploadButton.config(font=font1)

pathlabel = tk.Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)          
pathlabel.place(x=250, y=550)

splitButton = tk.Button(main, text="Split Dataset", command=splitdataset, bg="light green", width=15)
splitButton.place(x=450, y=600)
splitButton.config(font=font1)

LogesticRegression = tk.Button(main, text="LogesticRegression", command=LR, bg="lightgrey", width=15)
LogesticRegression.place(x=50, y=650)
LogesticRegression.config(font=font1)

random_forest = tk.Button(main, text="random_forest", command=random_forest, bg="pink", width=15)
random_forest.place(x=250, y=650)
random_forest.config(font=font1)

Run_DT = tk.Button(main, text="Run_DT", command=Run_DT, bg="yellow", width=15)
Run_DT.place(x=450, y=650)
Run_DT.config(font=font1)

Run_SVM = tk.Button(main, text="Run_SVM", command=Run_SVM, bg="lightgreen", width=15)
Run_SVM.place(x=650, y=650)
Run_SVM.config(font=font1)



pie_cart = tk.Button(main, text="pie_chart", command=pie_chart, bg="orange", width=15)
pie_cart.place(x=650, y=600)
pie_cart.config(font=font1)

plotButton = tk.Button(main, text="Plot Results", bg="lightgrey", width=15, command=plot_results)
plotButton.place(x=850, y=650)
plotButton.config(font=font1)

predict_button = tk.Button(main, text="Prediction", command=predict, bg="orange", width=15)
predict_button.place(x=1050, y=650)
predict_button.config(font=font1)

main.config(bg='#32d1a7')
main.mainloop()
