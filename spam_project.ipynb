import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

###

data = pd.read_csv('C:/Users/M/Desktop/Programming/spam.csv')
data.Category.replace({'ham' : 0 , 'spam' : 1},inplace=True)
data.head()

###

stemmer = PorterStemmer()
cops = []

for i in range(len(data.Message)):
    Part = re.sub('[^a-zA-z]' , ' ' , data.Message[i])
    Part = ' '.join([stemmer.stem(word) for word in  Part.lower().split(" ") if word not in set(stopwords.words('english'))])
    cops.append(Part)

###

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(cops).toarray()
Y = data.Category

###

x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.2)

###

model1 = MultinomialNB().fit(x_train , y_train)                 # A good model for TF-IDF
y_pre1 = model1.predict(x_test)
print('Model accuracy :' , accuracy_score(y_test , y_pre1))

###

model2 = RandomForestClassifier(n_estimators=100 , max_depth=100).fit(x_train , y_train)
y_pre2 = model2.predict(x_test)
print('Model accuracy :' , accuracy_score(y_test , y_pre2))
