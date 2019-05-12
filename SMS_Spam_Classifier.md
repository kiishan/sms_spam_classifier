
# SMS Spam Classifcation

**Context**

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. 
It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

**Data**

The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.

**Library Used**

- pandas
- numpy
- nltk
- sklearn
---

# Approach

- Loading Data
- Input and Output Data
- Applying Regular Expression
- Each word to lower case
- Splitting words to Tokenize
- Stemming with PorterStemmer handling Stop Words
- Preparing Messages with Remaining Tokens
- Preparing WordVector Corpus
- Applying Classification
---

## Importing Libraries


```python
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
```

---
## Loading Data


```python
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1</th>
      <th>v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
  </tbody>
</table>
</div>



---
## Understanding Data


```python
# Replace ham with 0 and spam with 1

df = df.replace(['ham','spam'],[0, 1]) 
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1</th>
      <th>v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Total ham(0) and spam(1) messages
df['v1'].value_counts()
```




    0    4825
    1     747
    Name: v1, dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5572 entries, 0 to 5571
    Data columns (total 2 columns):
    v1    5572 non-null int64
    v2    5572 non-null object
    dtypes: int64(1), object(1)
    memory usage: 87.1+ KB
    

#### Count the number of words in each Text


```python
df['Count']=0
for i in np.arange(0,len(df.v2)):
    df.loc[i,'Count'] = len(df.loc[i,'v2'])
    
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1</th>
      <th>v2</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
    </tr>
  </tbody>
</table>
</div>




```python
corpus = []
ps = PorterStemmer()
```


```python
# Original Messages

print (df['v2'][0])
print (df['v2'][1])
```

    Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
    Ok lar... Joking wif u oni...
    

---
## Processing Messages


```python
for i in range(0, 5572):

    # Applying Regular Expression
    
    '''
    Replace email addresses with 'emailaddr'
    Replace URLs with 'httpaddr'
    Replace money symbols with 'moneysymb'
    Replace phone numbers with 'phonenumbr'
    Replace numbers with 'numbr'
    '''
    msg = df['v2'][i]
    msg = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', df['v2'][i])
    msg = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', df['v2'][i])
    msg = re.sub('£|\$', 'moneysymb', df['v2'][i])
    msg = re.sub('\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', df['v2'][i])
    msg = re.sub('\d+(\.\d+)?', 'numbr', df['v2'][i])
    
    ''' Remove all punctuations '''
    msg = re.sub('[^\w\d\s]', ' ', df['v2'][i])
    
    if i<2:
        print("\t\t\t\t MESSAGE ", i)
    
    if i<2:
        print("\n After Regular Expression - Message ", i, " : ", msg)
    
    # Each word to lower case
    msg = msg.lower()    
    if i<2:
        print("\n Lower case Message ", i, " : ", msg)
    
    # Splitting words to Tokenize
    msg = msg.split()    
    if i<2:
        print("\n After Splitting - Message ", i, " : ", msg)
    
    # Stemming with PorterStemmer handling Stop Words
    msg = [ps.stem(word) for word in msg if not word in set(stopwords.words('english'))]
    if i<2:
        print("\n After Stemming - Message ", i, " : ", msg)
    
    # preparing Messages with Remaining Tokens
    msg = ' '.join(msg)
    if i<2:
        print("\n Final Prepared - Message ", i, " : ", msg, "\n\n")
    
    # Preparing WordVector Corpus
    corpus.append(msg)
```

    				 MESSAGE  0
    
     After Regular Expression - Message  0  :  Go until jurong point  crazy   Available only in bugis n great world la e buffet    Cine there got amore wat   
    
     Lower case Message  0  :  go until jurong point  crazy   available only in bugis n great world la e buffet    cine there got amore wat   
    
     After Splitting - Message  0  :  ['go', 'until', 'jurong', 'point', 'crazy', 'available', 'only', 'in', 'bugis', 'n', 'great', 'world', 'la', 'e', 'buffet', 'cine', 'there', 'got', 'amore', 'wat']
    
     After Stemming - Message  0  :  ['go', 'jurong', 'point', 'crazi', 'avail', 'bugi', 'n', 'great', 'world', 'la', 'e', 'buffet', 'cine', 'got', 'amor', 'wat']
    
     Final Prepared - Message  0  :  go jurong point crazi avail bugi n great world la e buffet cine got amor wat 
    
    
    				 MESSAGE  1
    
     After Regular Expression - Message  1  :  Ok lar    Joking wif u oni   
    
     Lower case Message  1  :  ok lar    joking wif u oni   
    
     After Splitting - Message  1  :  ['ok', 'lar', 'joking', 'wif', 'u', 'oni']
    
     After Stemming - Message  1  :  ['ok', 'lar', 'joke', 'wif', 'u', 'oni']
    
     Final Prepared - Message  1  :  ok lar joke wif u oni 
    
    
    


```python
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
```

---
# Applying Classification

- Input : Prepared Sparse Matrix
- Ouput : Labels (Spam or Ham)


```python
y = df['v1']
print (y.value_counts())
```

    0    4825
    1     747
    Name: v1, dtype: int64
    

### Encoding Labels


```python
le = LabelEncoder()
y = le.fit_transform(y)
```

### Splitting to Training and Testing DATA


```python
xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size= 0.20, random_state = 0)
```

---
## Applying Guassian Naive Bayes


```python
bayes_classifier = GaussianNB()
bayes_classifier.fit(xtrain, ytrain)
```




    GaussianNB(priors=None, var_smoothing=1e-09)




```python
# Predicting
y_pred = bayes_classifier.predict(xtest)
```

## Results


```python
# Evaluating
cm = confusion_matrix(ytest, y_pred)
```


```python
cm
```




    array([[824, 125],
           [ 19, 147]], dtype=int64)




```python
print ("Accuracy : %0.5f \n\n" % accuracy_score(ytest, bayes_classifier.predict(xtest)))
print (classification_report(ytest, bayes_classifier.predict(xtest)))
```

    Accuracy : 0.87085 
    
    
                  precision    recall  f1-score   support
    
               0       0.98      0.87      0.92       949
               1       0.54      0.89      0.67       166
    
        accuracy                           0.87      1115
       macro avg       0.76      0.88      0.80      1115
    weighted avg       0.91      0.87      0.88      1115
    
    

---
## Applying Decision Tree


```python
dt = DecisionTreeClassifier(random_state=50)
dt.fit(xtrain, ytrain)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=50, splitter='best')




```python
# Predicting
y_pred_dt = dt.predict(xtest)
```

## Results


```python
# Evaluating
cm = confusion_matrix(ytest, y_pred_dt)

print(cm)
```

    [[943   6]
     [ 29 137]]
    


```python
print ("Accuracy : %0.5f \n\n" % accuracy_score(ytest, dt.predict(xtest)))
print (classification_report(ytest, dt.predict(xtest)))
```

    Accuracy : 0.96861 
    
    
                  precision    recall  f1-score   support
    
               0       0.97      0.99      0.98       949
               1       0.96      0.83      0.89       166
    
        accuracy                           0.97      1115
       macro avg       0.96      0.91      0.93      1115
    weighted avg       0.97      0.97      0.97      1115
    
    

# Final Accuracy

- **Decision Tree : 96.861%**
- **Guassian NB   : 87.085%**   
