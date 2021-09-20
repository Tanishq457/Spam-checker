import pandas as pd
import pickle


import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import GridSearchCV



# Loading the dataset
df = pd.read_csv('data.csv', sep='\t', names=['label', 'message'])

# Preprocessing
corpus = []
ps = PorterStemmer()

for i in range(0,df.shape[0]):

  # Cleaning special character from the message
  message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df.message[i])

  # Converting message into lower case
  message = message.lower()

  # Tokenizing by words
  words = message.split()

  # Removing stop words
  words = [word for word in words if word not in set(stopwords.words('english'))]

  # Stemming
  words = [ps.stem(word) for word in words]

  # Joining stemmed words
  message = ' '.join(words)


  corpus.append(message)
  
# Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(df['label'])
y = y.iloc[:, 1].values

pickle.dump(cv, open('cv-transform.pkl', 'wb'))



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB(alpha=0.3)
classifier.fit(X_train, y_train)

filename = 'model.pkl'
pickle.dump(classifier, open(filename, 'wb'))