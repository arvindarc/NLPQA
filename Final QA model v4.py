from allennlp.predictors.predictor import Predictor
import warnings
import sys
import re
import docx2txt
from nltk.corpus import stopwords
import nltk
import gensim
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  accuracy_score
from sklearn.cluster import KMeans
import datefinder
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

import pandas as pd
import numpy as np
from scipy import spatial
import flask
import os
import io
from flask import request,redirect,url_for
from pymongo import MongoClient
import platform
import time

#Model - bidaf model
predictor = Predictor.from_path("./bidaf-model-2017.09.15-charpad")


if not sys.warnoptions:
    warnings.simplefilter("ignore")



def get_file(file_name,query):

    print("inside model:",file_name)
    ext = file_name.split(".")[-1]
    text = ''
    print("Found file with extension "+ ext)

    if ext == 'docx':
        text = docx2txt.process(file_name)

    elif ext == 'txt':
        with open(file_name) as f:
            for line in f:
                text = text+line

    elif ext == 'xlsx':
        f = pd.ExcelFile(file_name)
        for names in f.sheet_names:
            sheet = pd.read_excel(f,names,header = None)
            for row in sheet.values:
                for w in row:
                    text = text + str(w)

    elif ext ==  'pdf':
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle)
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
        with open(file_name, 'rb') as fh:
            for page in PDFPage.get_pages(fh,
                                          caching=True,
                                          check_extractable=True):
                page_interpreter.process_page(page)
            text = fake_file_handle.getvalue()
        converter.close()
        fake_file_handle.close()

    print(ext,len(text),type(text))

    train_data = pd.read_csv("traindata.csv")

    sentences = re.split('\n',text)

    dataset_sentences=pd.DataFrame(sentences,columns=["sentences"])

    null_sentences=dataset_sentences["sentences"]!=''
    dataset_sentences=dataset_sentences[null_sentences]
    final_sentence = []
    for sent in dataset_sentences["sentences"]:
        final_sentence.append(sent.lstrip('0123456789. '))

    final_df = pd.DataFrame(final_sentence,columns=["final_sentences"])

    final_df["final_sentences"] = final_df["final_sentences"].str.replace('"','')
    punctuations = list("!:?.;,_%`()")
    for punct in punctuations:
        final_df["final_sentences"] = final_df["final_sentences"].str.replace(punct,'')

    final_df["final_sentences"] = final_df["final_sentences"].str.replace("’s",'')

    punctuations2 = list("\-/")
    for punct2 in punctuations2:
        final_df["final_sentences"] = final_df["final_sentences"].str.replace(punct2,' ')
    for i in range(2):
        final_df["final_sentences"] = final_df["final_sentences"].str.replace("  ",' ')

    final_df["final_sentences"] = final_df["final_sentences"].str.lower()

    stop_words = list(stopwords.words('english'))
    stopwords_1 = ["would","able","due","one","need","co","so4","socio","many","small","low","go","per"]
    stopwords_final = stop_words + stopwords_1
    key_words = []


    for sentence in final_df["final_sentences"]:
        words = word_tokenize(sentence)
        for word in words:
            if word not in stopwords_final:
                key_words.append(word)

    lemmat = WordNetLemmatizer()
    lem_list = [lemmat.lemmatize(word,pos='v') for word in key_words]

    tag = nltk.pos_tag(lem_list)
    exclude_tag = ["RBR","RB","JJS","IN","CD","JJR","NNP","VBG","MD","CC","VBD","DT","VBN"]
    tagged_list = []
    [tagged_list.append(x[0]) for x in tag if x[1] not in exclude_tag]

    keywords_d = []
    [keywords_d.append(x) for x in tagged_list if x not in keywords_d]
    keywords_df = pd.DataFrame(keywords_d,columns=['keywords'])

    vector = Word2Vec([keywords_d],min_count=1)
    vector_all =[]
    for x in keywords_d:
        vector_all.append(vector[x].tolist())

    X_train = list(train_data["keywords"])
    y_train = list(train_data["prediction_numeric"])

    vector1 = Word2Vec([X_train], min_count=1)
    vector_train1 = []
    for x in X_train:
        vector_train1.append(vector1[x].tolist())

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(vector_train1,y_train)

    keywords_df["prediction"] = knn.predict(vector_all)

    keywords_df["prediction_word"] = np.where(keywords_df["prediction"]==1, "customer",
                                              np.where(keywords_df["prediction"]==2, "employee",
                                              np.where(keywords_df["prediction"]==3, "finance",
                                              np.where(keywords_df["prediction"]==4, "industry","management"))))

    final_text = ""
    for sent in final_sentence:
        final_text+= sent+" "


    def get_topic(query):
        q_tokens = word_tokenize(query)
        q_tokens_pos = nltk.pos_tag(q_tokens)
        exclude_tag = ["RBR","JJS","IN","CD","JJR","NNP","VBG","MD","CC","VBD","DT","VBN","VBZ","WP",'.']
        q_tagged_list = []
        [q_tagged_list.append(x[0]) for x in q_tokens_pos if x[1] not in exclude_tag]

        topic = []

        for query_word in q_tagged_list:
            pred = keywords_df.loc[keywords_df["keywords"] == query_word]
            for i in pred["prediction_word"]:
                if i!=0:
                    if i not in topic:
                        topic.append(i)

        return topic

    def main_query(query):
        actual_query = query
        query = query.replace('?','')
        new_text = ""
        new_sentences = ""
        new1 = ""

        if ext == "docx":
            passage = docx2txt.process(file_name)
            sentences = re.split('\n',passage)
            new_text = ""
            for i in sentences:
                if i!="":
                    j = i.lstrip('0123456789. ')
                    if (len(j) != len(i)):
                        if new_text!="":
                            new_text = new_text+" "+j
                        else:
                            new_text = new_text+j
            new1 =new_text
            new_sentences = sent_tokenize(new_text)
            print('inside docx')

        elif ext == 'txt':
            passage = ""
            with open(file_name) as f:
                        for line in f:
                            passage = passage+line
            sentences = re.split('\n',passage)
            new_text = ""
            print("Length of sentences generated :",len(sentences))
            for i in sentences:
                if i!="":
                    j = i.lstrip('0123456789. ')
                    if (len(j) != len(i)):
                        if new_text!="":
                            new_text = new_text+" "+j
                        else:
                            new_text = new_text+j

            new_sentences = sent_tokenize(new_text)
            print('inside txt')

        elif ext == 'pdf':
            text = ""
            resource_manager = PDFResourceManager()
            fake_file_handle = io.StringIO()
            converter = TextConverter(resource_manager, fake_file_handle)
            page_interpreter = PDFPageInterpreter(resource_manager, converter)
            with open(file_name, 'rb') as fh:
                for page in PDFPage.get_pages(fh,
                                              caching=True,
                                              check_extractable=True):
                    page_interpreter.process_page(page)
                text = fake_file_handle.getvalue()
            converter.close()
            fake_file_handle.close()
            passage1 = text
            print("PDF")
            text_split = passage1.split()
            pdf_sent = ""
            for t in text_split:
                t = t.lstrip('0123456789. ')
                if t != "":
                    if pdf_sent =="":
                        pdf_sent = t +" " +pdf_sent
                    else:
                        pdf_sent = pdf_sent +" " +t

            print(pdf_sent)

            new_sentences = sent_tokenize(pdf_sent)
            print("PDF tokenize: ",len(new_sentences))

            new_text = ""
            for sent in new_sentences:
                new_text = sent+new_text

            print('inside pdf')

        elif ext == "xlsx":
            text = ""
            f = pd.ExcelFile(file_name)
            for names in f.sheet_names:
                sheet = pd.read_excel(f,names,header = None)
                for row in sheet.values:
                    for w in row:
                        w = w.lstrip('0123456789. ')
                        if text =="":
                            text = text + str(w)
                        else:
                            text = text + " " + str(w)

            new_text= text
            new_sentences = sent_tokenize(new_text)
            print("xlsx tokenize: ",len(new_sentences))
            print('inside excel')

        new2 = new_text
        print(new1==new2)
        print(new_text)
        print(len(new_text))




        if query.startswith('is') or query.startswith('does'):

            result=predictor.predict(passage=new_text,question=query)
            answer= result['best_span_str']

            tokenized_doc = []

            for d in final_df["final_sentences"]:
                tokenized_doc.append(word_tokenize(d.lower()))

            tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]

            model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, epochs = 100)
            model.save("test_doc2vec.model")
            model= Doc2Vec.load("test_doc2vec.model")

            q_tokens = word_tokenize(query)
            q_tokens_pos = nltk.pos_tag(q_tokens)
            exclude_tag = ["RBR","JJS","IN","CD","JJR","NNP","VBG","MD","CC","VBD","DT","VBN","VBZ"]
            q_tagged_list = []
            [q_tagged_list.append(x[0]) for x in q_tokens_pos if x[1] not in exclude_tag]

            a_tokens = word_tokenize(answer)
            a_tokens_pos = nltk.pos_tag(a_tokens)
            exclude_tag = ["RBR","JJS","IN","CD","JJR","NNP","VBG","MD","CC","VBD","DT","VBN","VBZ"]
            a_tagged_list = []
            [a_tagged_list.append(x[0]) for x in a_tokens_pos if x[1] not in exclude_tag]

            query_final = ""
            for i in q_tagged_list:
                query_final+= i+" "

            answer_final = ""
            for i in a_tagged_list:
                answer_final+= i+" "


            vec1 = model.infer_vector(query_final.split())
            vec2 = model.infer_vector(answer_final.split())

            similairty = spatial.distance.cosine(vec1, vec2)

            if ((similairty >= 0.005 and similairty <= 0.006) or (similairty >= 0.012 and similairty <= 0.022) or (similairty >= 0.0561 and similairty <= 0.0568)):
                return "No"

            else:
                return "Yes"

        else:

            if actual_query.endswith("?"):
                actual_query=actual_query
            else:
                actual_query = actual_query + "?"

            result=predictor.predict(passage=new_text,question=actual_query)
            answer = result['best_span_str']
            similarity_value =[]
            print(len(new_sentences))
            print('inside what questions : ')
            print(answer)
            for k in new_sentences:

                output_tokenize = word_tokenize(answer)
                k_tokenize = word_tokenize(k)

                sw = stopwords.words('english')
                l1=[];l2 =[]

                output_set = {w for w in output_tokenize if not w in sw}
                k_set = {w for w in k_tokenize if not w in sw}

                rvector = output_set.union(k_set)
                for w in rvector:
                    if w in output_set: l1.append(1) # create a vector
                    else: l1.append(0)
                    if w in k_set: l2.append(1)
                    else: l2.append(0)
                c = 0

                for i in range(len(rvector)):
                    c+= l1[i]*l2[i]
                    cosine = c / float((sum(l1)*sum(l2))**0.5)

                similarity_value.append(cosine)


            print("Result : ")

            print(max(similarity_value))
            print(new_sentences[similarity_value.index(max(similarity_value))])

            answer = new_sentences[similarity_value.index(max(similarity_value))]

            return answer

    def datatype(query):

        Descriptive = ['what','which','who','whom','whose','why','where','how']
        Number = ['how much','how many','how old','how far']
        Time = ['when','how long']
        Boolean = ['is','does']
        secondary_word = ['profit','sum','mean','percentage','total','loss','difference','age','average','maximum','minimum']

        query_words = word_tokenize(query)
        query_first_word = query_words[0]
        query_second_word = query_words[1]
        query_both_words = query_first_word+" "+query_second_word

        i=0
        for w in query_words[1:]:
            if w in secondary_word:
                i+=1
        if query_first_word =='what' and i > 0:
            ans_type= 'Numerical'
        elif query_both_words in Number:
            ans_type= 'Numerical'
        elif query_first_word in Time or query_both_words in Time:
            ans_type = 'Date/Time'
        elif query_first_word in Descriptive:
            ans_type = 'Text'
        elif query_first_word in Boolean:
            ans_type = 'Boolean'
        else:
            ans_type = 'Please enter valid question'

        return ans_type

    return main_query(query),get_topic(query),datatype(query)

# Flask module begins

client = MongoClient("mongodb://localhost")
db=client["DATABASE_NAME"]
collection=db["USERINTERFACE2"]
comp_list = []
name=''
count = collection.count_documents({})
print(count,"count")
comp_dict={}
app = flask.Flask(__name__)

@app.route('/')
def frontpage():

    comp_list = []
    for x in collection.find({},{ "Company Name": 1}):
        comp = x['Company Name']
        comp = comp.replace("./Upload/","")
        comp = comp.split(".")[0]
        comp_list.append(comp)

    return flask.render_template('final.html',company_list = comp_list)


@app.route('/handleUpload',methods = ['POST', 'GET'])
def handleUpload():

        if 'browse' in request.files:
            browse = request.files['browse']

            #browse module begins

            if browse.filename != '':
                print('inside browse function')

                filename = str(browse.filename)
                name = filename.split('.')[0]
                print('name:',name)

                browse.save(os.path.join('./Upload', browse.filename))
                path_name = './Upload/'+ browse.filename


                def creation_date(path_to_file):
                    if platform.system() == 'Windows':
                        return os.path.getctime(path_to_file)
                    else:
                        stat = os.stat(path_to_file)
                        try:
                            return stat.st_birthtime,stat.st_mtime
                        except AttributeError:
                            return stat.st_mtime

                date_out = creation_date(os.path.join('./Upload', browse.filename))
                from datetime import datetime

                timestamp = 1528797322
                date_time = datetime.fromtimestamp(timestamp)

            #print("Date time object:", date_time)

                date_out= date_time.strftime("%m/%d/%Y, %H:%M:%S")
                comp_list = []
                for x in collection.find({},{"Company Name": 1}):
                    comp = x['Company Name']
                    comp = comp.replace("./Upload/","")
                    comp = comp.split(".")[0]
                    comp_list.append(comp)

                comp_id = []
                for y in collection.find({},{"CompanyId": 1}):
                    comp_id.append(y["CompanyId"])

                print(type(1))
                if name not in comp_list:
                    if count==0:
                        record = {'CompanyId':1,'Company Name':path_name,'Created Date':date_out}
                        collection.insert(record)
                    else:
                        max_compid = int(np.max(comp_id)+1)
                        print(max_compid,type(max_compid))
                        record = {'CompanyId':max_compid,'Company Name':path_name,'Created Date':date_out}
                        collection.insert(record)

                    return redirect(url_for('frontpage'))
                else:
                    return flask.render_template('Company.html',company_list = comp_list)


            else:
                #drop down module begins

                if count==0:
                    return redirect(url_for('frontpage'))
                else:
                    company_name = request.form.to_dict()
                    print(company_name,type(company_name),company_name.keys())
                    get_company=company_name['companies']
                    print(get_company)

                    comp_list=[]
                    for x in collection.find({},{ "Company Name": 1}):
                        comp = x['Company Name']
                        comp = comp.replace("./Upload/","")
                        comp = comp.split(".")[0]
                        comp_list.append(comp)

                    return flask.render_template('Company.html',company_list = comp_list)

@app.route('/companydetails',methods = ['POST', 'GET'])

def companydetails():

        print('inside companydetails')
        if count==0:
            print('count')
            return redirect(url_for('frontpage'))
        else:
            print('2nd dropdown')
            company_name = request.form.to_dict()
            print(company_name,type(company_name),company_name.keys())
            get_company=company_name['companies_2']
            print(get_company)

            if flask.request.method == 'POST':
              result = flask.request.form.to_dict()
              print(result,"result")
              get_question=result['The Entered Keyword is']

              comp_list = []


              for x in collection.find({},{ "Company Name": 1}):
                comp_ori = x['Company Name']
                comp = comp_ori.replace("./Upload/","")
                comp1 = comp.split(".")[0]
                comp_list.append(comp1)
                if comp1 not in comp_dict:
                    comp_dict[comp1] = comp_ori

              print(comp_dict)


              print('get_company',get_company)

              file_name = comp_dict[get_company]
              a,b,c = get_file(file_name,get_question.lower())

              client = MongoClient("mongodb://localhost")
              db=client["DATABASE_NAME"]
              collection2=db["ANSWERS"]
              rec1={'list of topics': b, 'question' : get_question ,'Answer' : a,'Answer datatype' : c }
              lll=collection2.insert_one(rec1)



              return flask.render_template("Company.html",topic=b,query=get_question,result=a,dtype =c,company_list = comp_list)


if __name__ == '__main__':
   app.run(debug = True,use_reloader=False)