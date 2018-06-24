# Run  code on jupyter notebook
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os

# series:       1 dimensional labelled / indexed array
# dataframes:   A dataframe is similar to Excel workbook
#              – you have column names referring to columns and you have rows,
#                which can be accessed with use of row numbers

data_file = os.path.join("train_loan_pred.csv")

train_data = pd.read_csv(data_file)
# to get general idea of data-fileds, first 10 examples are displayed
train_data.head(10)

# to get additional information about the dataset eg , missing values, percentages, means
#	         ApplicantIncome	  CoapplicantIncome	  LoanAmount	Loan_Amount_Term	Credit_History
# ========================================================================================
#   count	614.000000	              614.000000	592.000000	      600.00000	      564.000000
#   mean	5403.459283	              1621.245798	146.412162	      342.00000	      0.842199
#   std	    6109.041673	              2926.248369	85.587325	      65.12041	      0.364878
#   min	    150.000000	              0.000000	    9.000000	      12.00000	      0.000000
#   max	    81000.000000	          41667.000000	700.000000	     480.00000	      1.000000
train_data.describe()
# to get statistics of non numerical fileds
#   Graduate        480
#   Not Graduate    134
#   Name: Education, dtype: int64
train_data['Education'].value_counts()

# to actualy analyse the data after this point various graphs, plots can be used
# eg :  histogram for genral idea
#       box plot / group box plot for distribution-outlier clarity
train_data['ApplicantIncome'].hist(bins=100)
train_data.boxplot(column='ApplicantIncome')
train_data.boxplot(column='ApplicantIncome', by = 'Education')
train_data.boxplot(column='ApplicantIncome', by = 'Loan_Status')
