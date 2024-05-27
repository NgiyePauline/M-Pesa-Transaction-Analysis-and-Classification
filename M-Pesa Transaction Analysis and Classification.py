import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from PyPDF2 import PdfReader, PdfWriter

#Decrypt pdf file (mpesa statement)
def remove_pdf_password(input_path, output_path, password):
    try:
        #Open the input PDF
        with open(input_path, 'rb') as input_file:
            pdf_reader = PdfReader(input_file)
            
            if pdf_reader.is_encrypted:
                print("PDF is encrypted. Attempting to decrypt")
                pdf_reader.decrypt(password)
                print("File decrypted successfully.")
            else:
                print("PDF is not encrypted.")
            
            #Create a PdfWriter object
            pdf_writer = PdfWriter()
            
            #Add pages to the writer
            for page_num in range(len(pdf_reader.pages)):
                pdf_writer.add_page(pdf_reader.pages[page_num])
                print(f"Added page {page_num + 1}")
            
            #Write the decrypted content to the output file
            with open(output_path, 'wb') as output_file:
                pdf_writer.write(output_file)
                
            print("Decrypted PDF saved successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

#What to use
input_path = 'MPESA_Statement.pdf'
output_path = 'MyStatement.pdf'
password = '183177'

remove_pdf_password(input_path, output_path, password)

DocPath = output_path

#Extract Transaction Table from pdf
def ExtractTable(pdf_path):
    all_rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                all_rows.extend(table)
    return all_rows

pdf_path = 'MyStatement.pdf'

#Merge extracted tables to one single table (tables extracted from each pdf page)
merged_table = ExtractTable(pdf_path)

#Create a dataframe
df = pd.DataFrame(merged_table)
unwanted_keywords = ['Receipt No', 'Details', 'Paid In']

#Data CleanUp
#Loop through each row and column in the DataFrame to remove unwanted data
for index, row in df.iterrows():
    for column in df.columns:
        for keyword in unwanted_keywords:
            if keyword in str(row[column]):
                #Drop the row if any unwanted keyword is found in any column
                df.drop(index, inplace=True)
                break  
        else:
            continue  
        break

headers = ['Receipt No', 'Completion Time', 'Details', 'Transaction Status', 'Paid In', 'Withdrawn', 'Balance']
df.columns = headers  
print(df.head(2))
row_count = len(df)
print(f'The dataframe row count is: {row_count}')

#Write the dataframe to a CSV file
df.to_csv("CleanData.csv", index=False)

#Process the data
#Load the csv file
CleanDf = pd.read_csv('CleanData.csv')

#Ensure 'Completion Time' is datetime
CleanDf['Completion Time'] = pd.to_datetime(CleanDf['Completion Time'], errors='coerce')

#Convert numeric columns from strings with commas to float
for col in ['Paid In', 'Withdrawn', 'Balance']:
    CleanDf[col] = CleanDf[col].str.replace(',', '').astype(float)

#Show dataset Info
print(CleanDf.info())

#Show Total number of null values
print(CleanDf.isnull().sum())

#Show Datatypes
print(CleanDf.dtypes)

#Processing the data
#Fill empty cells with 0 in 'Withdrawal' and 'Paid In' columns
CleanDf['Withdrawn'] = CleanDf['Withdrawn'].fillna(0)
CleanDf['Paid In'] = CleanDf['Paid In'].fillna(0)

#Convert 'Withdrawn' and 'Paid In' columns to numeric
CleanDf['Withdrawn'] = pd.to_numeric(CleanDf['Withdrawn'], errors='coerce')
CleanDf['Paid In'] = pd.to_numeric(CleanDf['Paid In'], errors='coerce')

#Visualize the data
#Creating a pairplot 
columns_to_plot = ['Withdrawn', 'Paid In', 'Balance', 'Transaction Status']
pairplot_data = CleanDf[columns_to_plot]

sns.pairplot(pairplot_data, hue='Transaction Status', height=3, palette='Set1')
plt.show()

#Line graph to show Paid In and Withdrawn over time
plt.figure(figsize=(14, 7))
plt.plot(CleanDf['Completion Time'], CleanDf['Paid In'], label='Paid In', color='green')
plt.plot(CleanDf['Completion Time'], CleanDf['Withdrawn'], label='Withdrawn', color='red')
plt.title('Paid In and Withdrawn Over Time')
plt.xlabel('Completion Time')
plt.ylabel('Amount')
plt.legend()
plt.grid(True)
plt.show()

#Build a regression model
#Select features and target variable
features = CleanDf[['Paid In', 'Balance']]
target = CleanDf['Withdrawn']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Initialize and train the regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

#Make predictions on the testing set
lr_pred = lr_model.predict(X_test)

#Evaluate the Linear Regression model
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print(f'Linear Regression Mean Absolute Error: {lr_mae}')
print(f'Linear Regression Mean Squared Error: {lr_mse}')
print(f'Linear Regression R-squared: {lr_r2}')

#Cross-Validation for Linear Regression
lr_cv_scores = cross_val_score(lr_model, features, target, cv=5, scoring='r2')
print(f'Linear Regression Cross-validated R-squared scores: {lr_cv_scores}')
print(f'Linear Regression Mean Cross-validated R-squared: {lr_cv_scores.mean()}')

#Initialize and train the RandomForest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

#Make predictions on the testing set using RandomForest
rf_pred = rf_model.predict(X_test)

#Evaluate the RandomForest model
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f'Random Forest Mean Absolute Error: {rf_mae}')
print(f'Random Forest Mean Squared Error: {rf_mse}')
print(f'Random Forest R-squared: {rf_r2}')

#Cross-Validation for RandomForest
rf_cv_scores = cross_val_score(rf_model, features, target, cv=5, scoring='r2')
print(f'Random Forest Cross-validated R-squared scores: {rf_cv_scores}')
print(f'Random Forest Mean Cross-validated R-squared: {rf_cv_scores.mean()}')

#Feature Importance
feature_importance = pd.Series(rf_model.feature_importances_, index=features.columns).sort_values(ascending=False)
print("Feature Importance from RandomForest:")
print(feature_importance)

#Plot to show the true vs predicted values for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, lr_pred, alpha=0.7, label='Linear Regression Predictions')
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Withdrawn Amounts (Linear Regression)')
plt.legend()
plt.grid(True)
plt.show()

#Plot to show the true vs predicted values for RandomForest
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_pred, alpha=0.7, label='RandomForest Predictions')
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Withdrawn Amounts (RandomForest)')
plt.legend()
plt.grid(True)
plt.show()

#Save the cleaned DataFrame to the existing CSV file
CleanDf.to_csv("CleanData.csv", index=False)
