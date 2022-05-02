# We will write the program in total in 3 parts -

# Part1 :- Data Preprocessing 
# Step1 - Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Step2 - Importing Dataset
dataset = pd.read_csv('Churn_Modelling.csv');

#Step3 - Defining feature matrix(X) and Target array(y) 
X = dataset.iloc[:,  3:13].values;
y = dataset.iloc[:, 13].values;

#Step4 - Encoding categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X_1 = LabelEncoder() 
X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])

label_encoder_X_2 = LabelEncoder() 
X[:, 2] = label_encoder_X_1.fit_transform(X[:, 2])

# Step6 - One hot encoding previously encoded categorical data
"""
We have encoded Geography as well as Gender but one hot encoding is needed 
only for Geography as Gender have only two possible values so it is encoded in 
0 and 1 only.
""" 
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
   [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],  
   remainder='passthrough'                                         
)
X = ct.fit_transform(X)

#Step7 - Removing one dummy variable to avoid dummy variable trap
"""
In this step we will remove 1 column from newly created 3 columns in previous 
step using one hot encoding to avoid dummy variable trap
""" 
X= X[:, 1:]

# Step8 - Splitting data into training data and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=1)

# Step9 - Feature scaling training and test data for better performance 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Part2 :- Implementing ANN
# Step1 - importing Keras library and required packages.
"""
We are not importing tenserflow or keras library directly. Instead we are 
importing only those modules of keras which we require in this program. 
However if needed we can import tensorflow and keras as follows :
import tensorflow as tf
from tensorflow import keras  
"""
from tensorflow.keras.models import Sequential
# Sequential class is required to initialize neural network
from tensorflow.keras import Input
# Input class is used to define shape of input layer 
from tensorflow.keras.layers import Dense
# Dense class is required to create hidden layers in ANN

# Step2 - Initializing ANN
classifier = Sequential()
# Here classifier is nothinh but the ANN which we are going to build. 

# Step3 - Adding input layer
classifier.add(Input(shape=(11,)))

# Step4 - Adding first hidden layer
# Note :- Check ANN_steps.png(stored in current folder itself) file for 
# details about required steps.
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
"""
In Dense function first argument is "units" which is nothing but the number of 
nodes you want in the current hidden layer. 
Note that there is no thumb rule to choose optimal number of nodes in any 
hidden layer.You can follow some suggested techniques like you can experiment 
with a technique called "Parameter tuning" like "k-fold cross validation" etc.
However you can follow this tip which works optimally most of the time -
choose number of nodes as the average of number of nodes in input and output 
layers.Since our input layer consist of 11 nodes(check X_train) and output 
layer consists of 1 layer so its average is 6 and that's the reason we have 
taken unit as 6.    
Second needed argument is "kernel_initializer" it is the function which 
initializes 'weights' for every input nodes. we have choosen 'uniform' for it.
Third needed argument is "activation" we will choose 'rectifier function' as 
activation function for hidden layers and 'sigmoid function' as activation 
function in output layer. 'rectifier function' is denotd by 'relu' so we have 
choosen 'relu' here. 
"""
# Step5 - Adding second hidden layer(We will keep only 2 hidden layers)
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
"""
Here you can change number of nodes in this lyer by updating "units". We 
have choosen same as in previous layer as still average of input and output 
nodes are same. And since this is also a hidden layer so we choosen 
activation as 'relu'.
"""

# Step6 - Adding output layer
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
"""
Since output layer must have 1 node only so we choosed units as 1 and as 
discussed earlier also we will use 'sigmoid function' as activation function 
for output layer so we choosed 'sigmoid' here for 'activation'.
"""

# Step7 - Compiling ANN(i.e applying Stochastic Gradient Descent to the ANN)
classifier.compile(optimizer='adam', loss='binary_crossentropy', 
                   metrics=['accuracy'])
"""
There are several types of Stochastic Gradient Descent algorithm out there.
'adam' is one of them.So we choosed optimizer as 'adam'. Second parameter is 
'loss' which is nothing but asking about the 'loss function'. Since it is a 
classification problem so we will choose 'logarithmic loss function'. 
If your dependent variable is of type binary then the name for 'logarithmic 
loss function' is 'binary_crossentropy' and if the dependent variable is of 
type categorical(i.e have more than 2 possible outcomes) then this 'logarithmic 
loss function' is called 'categorical_crossentropy'. Since out output layer is 
of type 'binary'.So we choosed 'binary_crossentropy'.
Third parameter is 'matrics'. It isjust the creterion you choose to evaluate 
your model. When the weights are updated after each observation or after each 
batch of many observations the algorithm uses these creterions to improve the 
model's performance i.e every time weight is updated accuracy will be improved. 
We have choosen only 'accuracy' as creterion. But still we have to enter it 
in array format as 'metrics' requires data in array format.
"""

# Step 8 - Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)
"""
We can update the weight either after each observation or after a batch of 
observations. So here third argument is asking for the batch size after which we 
want to update the weight. 
And the fourth argument is epoch which is nothing but the number of repeatition 
we want to apply ANN on the training data.(Check ANN_steps.png in the current 
folder for details about epoch.) 
"""

# Part3 :- Making the predictions and evaluating the model
# Step1 :- Predicting the test set result
y_pred = classifier.predict(X_test)
"""
You can check the accuracy of each epoch from the epoch result. Epoch result 
will be in the following format :-
---------------------------------------------------------------------------
Epoch 81/100
800/800 [==============================] - 4s 5ms/step - loss: 0.3931 - 
accuracy: 0.8422
---------------------------------------------------------------------------
So here accuracy is 0.8422 i.e 84.22%
"""
y_pred = (y_pred > 0.5)
# i.e if y_pred > 0 then y_pred becomes true otherwise False

# Step2 :- Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
"""
cm will be in the following format :-
----------------------------
array([[1541,   44],
       [ 281,  134]])
----------------------------
If you want to check the accuracy of ANN on test set data we can calculate 
accuracy from the confusion metrix result as follows :
accuarcy = total currect predictions/ total predictions 
i.e in this case accuracy is : (1541+134)/2000 = 0.8375 or 83.75%
which can be considered as a good accuracy.
"""

