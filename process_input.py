import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np




lemmatizer=WordNetLemmatizer()

pickle_corpus=open("corpus.pkl","rb")
corpus=pickle.load(pickle_corpus)
pickle_cv=open("bag_of_words.pkl","rb")
cv=pickle.load(pickle_cv)


def process_sms_input(sms):
        data=re.sub('[^a-zA-z]',' ',sms)
        data=data.lower()                                               #lower case
        data=data.split()
        data=[lemmatizer.lemmatize(words) for words in data if words not in stopwords.words('english')]
        data=' '.join(data)
        corpus.append(data)
        x=cv.fit_transform(corpus).toarray()
        x_new=x[-1]
        print(x.shape)
        corpus.pop()
        x=np.delete(x,-1,axis=0)    #Recheck x against X
        print(x.shape)
        return x_new
    
    
       
       
               