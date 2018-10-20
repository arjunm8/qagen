#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from flask import Flask, render_template, jsonify,request
import pandas
import nltk
import numpy as np

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

reload(nltk)

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

app = Flask(__name__)

DEBUG = True

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response


@app.route("/get_qa")
def get_questions():

    #fetch passage via get req from frontend
    passage = str(request.args.get('passage'))

    #tokenize and tag the parts of speech.
    tagged = [nltk.pos_tag(nltk.word_tokenize(s)) for s in nltk.sent_tokenize(passage)]
    #Fetch PoS Rules for 'what' questions extracted from github
    sentence_structs = np.load("rules.npy")
    #define question array
    questions = []
    for sentence in tagged:
        #answer phrase or not
        nf = 1
        answer_phrase = pandas.DataFrame(sentence).set_index(1)
        #index statments according to pos
        statement = answer_phrase[~answer_phrase.index.duplicated(keep='first')]

        #detect answer phrase and generate qa pairs
        for sen_struc in sentence_structs[:8]:
            if all(rule_pos in [sen_pos[1] for sen_pos in sentence] for rule_pos in sen_struc[0]):
                q = "What " + " ".join( list(statement.loc[sen_struc[1]][0] ) ) + "?"
                nf=0
                break
        if nf and all(rule_pos in [sen_pos[1] for sen_pos in sentence] for rule_pos in sentence_structs[8]):
            if statement.loc['PRP'][0] in ['he','she']:
                q = "what does " + statement.loc['PRP'][0]+" " + str(statement.loc['VBZ'][0])[:-1] + "?"
                nf=0

        elif nf and all(rule_pos in [sen_pos[1] for sen_pos in sentence] for rule_pos in sentence_structs[9]):
            q = "what does " + statement.loc['NNP'][0]+" " + str(statement.loc['VBZ'][0])[:-1] + "?"
            nf=0

        if not nf:
            questions.append({
                            "question" : q,
                            "statement" : " ".join(list(answer_phrase[0]))
                            })
    #build a json a send to frontend
    return jsonify({'questions': questions})


@app.route("/")
def index_a():
    '''
    Temporarily hosting frontend on flask server as well so the requests are made to the same host
    '''

    return render_template("index.html")




if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
