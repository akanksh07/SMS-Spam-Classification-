# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 01:19:42 2021

@author: akanksh.belchada
"""
#pip install fastapi uvicorn
# 1. Library imports
import uvicorn ##ASGI       For Asynchronous Server Gateway Interface
from fastapi import FastAPI, Request

from fastapi.templating import Jinja2Templates



from SpamClassifier import Spam_Classifier_Model


templates=Jinja2Templates(directory="templates")
import pickle
import process_input
# 2. Create the app object
app = FastAPI()


# 3. Index route, opens automatically on http://127.0.0.1:8000

@app.get('/')
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})
# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
    

    
@app.post('/v1/authenticate_sms')
def classify_spam(temp:Spam_Classifier_Model):
    temp=temp.dict()
    data=temp['data']
    pickle_model=open("spam_detect_model.pkl","rb")
    spam_detect_model=pickle.load(pickle_model)
    processed_input=process_input.process_sms_input(data)
    prediction="SPAM" if spam_detect_model.predict([processed_input]) else "LEGIT"
    return {"The given SMS is" : prediction}


if __name__ == '__main__':
    print("Here")
    uvicorn.run(app, host='127.0.0.1', port=7000)
#uvicorn Handwriting_Recognition_api:app --reload

