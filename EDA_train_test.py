# import lib's
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import  r2_score
import matplotlib.pyplot as plt
pd.pandas.set_option('display.max_columns',None)

# read dataset
df= pd.read_csv('dataset/Admission_Predict.csv')

# Data Pre-processing and Exploratory Data Analysis
df.head()
""" 
   Serial No.  GRE Score  TOEFL Score  University Rating  SOP  LOR   CGPA  \
0           1        337          118                  4  4.5   4.5  9.65       
1           2        324          107                  4  4.0   4.5  8.87       
2           3        316          104                  3  3.0   3.5  8.00       
3           4        322          110                  3  3.5   2.5  8.67       
4           5        314          103                  2  2.0   3.0  8.21       

   Research  Chance of Admit
0         1              0.92
1         1              0.76
2         1              0.72
3         1              0.80
4         0              0.65 """

df.info()
""" 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 500 entries, 0 to 499
Data columns (total 9 columns):
 #   Column             Non-Null Count  Dtype
---  ------             --------------  -----
 0   Serial No.         500 non-null    int64
 1   GRE Score          500 non-null    int64
 2   TOEFL Score        500 non-null    int64
 3   University Rating  500 non-null    int64
 4   SOP                500 non-null    float64
 5   LOR                500 non-null    float64
 6   CGPA               500 non-null    float64
 7   Research           500 non-null    int64
 8   Chance of Admit    500 non-null    float64
dtypes: float64(4), int64(5)
memory usage: 35.3 KB """

# check for missing values 
df.isnull().sum()
""" 
Serial No.           0
GRE Score            0
TOEFL Score          0
University Rating    0
SOP                  0
LOR                  0
CGPA                 0
Research             0
Chance of Admit      0
dtype: int64 """

df.columns
""" 
Index(['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research', 'Chance of Admit '],
      dtype='object') """

# divide into independent and dependent features
# dropping the 'Chance of Admit' and 'serial number' as they are not going to be used as features for prediction
x = df.drop(['Chance of Admit ','Serial No.'],axis=1)

# 'Chance of Admit' is the target column which shows the probability of admission for a candidate
y=df['Chance of Admit ']

# Relationship between GRE Score and Chance of Admission
plt.scatter(df['GRE Score'],y) 
plt.show()

# Relationship between TOEFL Score and Chance of Admission
plt.scatter(df['TOEFL Score'],y) 
plt.show()

# # Relationship between CGPA and Chance of Admission
plt.scatter(df['CGPA'],y) 
plt.show()

# from above plots, there is a linear relationship between independent and dependent features
# so lets use linear regression model 

# splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.33,random_state=100)

# fitting the date to the Linear regression model
from sklearn import linear_model
reg = linear_model.LinearRegression()

reg.fit(train_x, train_y)
# LinearRegression()

# calculating the accuracy of the model
from sklearn.metrics import r2_score
score= r2_score(reg.predict(test_x),test_y)
# 0.8082585452743906

# saving the model to the local file system
filename = 'model/predict_student_admission_model.pickle'
pickle.dump(reg, open(filename, 'wb'))

# prediction using the saved model.
loaded_model = pickle.load(open(filename, 'rb'))

prediction=loaded_model.predict(([[320,120,5,5,5,10,1]]))
print(prediction[0])
# 0.9957864057903034

# for above input values, model predicts 99 % of student admission

# finally, lets go ahead to the main course i,e deployment