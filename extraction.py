from flask import Flask, request, render_template
import re
import requests
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions

# watson api for Keyword Extraction
authenticator = IAMAuthenticator('OxgvILpwXSEyaOUNrMMg7aQQeeB8lfoEqg-dJ-_T-boj')
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2021-03-25',
    authenticator=authenticator
)

natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/e29dd273-2c9a-4e3c-882f-7b98fe7053ea')



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/extractor')
def extractor():
    return render_template('extractor.html')

@app.route('/keywords', methods=['POST'])
def keywords():
    sen = request.form['output']
    type = request.form['type']
    # num = request.form['wordnum']
    keyword = check(sen,type)
    return render_template('keyword.html', keyword=keyword)

def check(sentence,type):
    # url = "https://textanalysis-keyword-extraction-v1.p.rapidapi.com/keyword-extractor-text"
    # payload = "text=" + sentence + "&wordnum=" + str(500)
    # headers = {
    #     'content-type': "application/x-www-form-urlencoded",
    #     'x-rapidapi-key': "bdf00f69d0mshf71b715e665de84p187496jsn103deb88c72c",
    #     'x-rapidapi-host': "textanalysis-keyword-extraction-v1.p.rapidapi.com"
    #     }
    #
    # response = requests.request("POST", url, data=payload, headers=headers)
    # print(response.text)

    if(type == "url"):
        response = natural_language_understanding.analyze(
            # url='www.ibm.com',
            url=sentence,
            features=Features(keywords=KeywordsOptions(sentiment=False, emotion=False))).get_result()
    else :
        response = natural_language_understanding.analyze(
            # url='www.ibm.com',
            text=sentence,
            features=Features(keywords=KeywordsOptions(sentiment=False, emotion=False))).get_result()
    k = []
    for i in response['keywords']:
        k.append(i['text'])
    return k

if __name__ == "__main__":
    app.run(debug="True")