
############################################################################
# May, 2022 
# The python script is to clean the data and embedded the glassdoor review
# then run topic classification
#
# Input: ./data/glassdoor_reviews_selected12.csv # glassdoor review csv 
#        ./model/  # pre-trained embedded model (downlaod from https://tfhub.dev/google/universal-sentence-encoder-multilingual/3)
#        ./topic/TwoCenturies_Glassdoor_topic_seedwords.xlsx  # defined topics' seedwords
# Output: ./output/*_probs.csv  # probs of each topic , of each review
#
#
# We process the following columns:
# Reviews: summary, pros_description, cons_description, advice_to_manager
# 
# This script is not run on multiple machine
# Step 1: Preprocessing, cleaning, and remove stop words
# Step 2: Sentence Embedding using pretrain model, we need to use the same model for topic embedding
# Step 3: Topic seedword embedding
# step 4: classification and output *_probs.csv to output dir
#
# by YuenYeeLo yylo7775@gmail.com
############################################################################


# global 
import global_options as glb
feat_cols=glb.feat_cols

# import library
import optparse
import pandas as pd
import numpy as np
from datetime import datetime
import os, re, string
from sklearn.feature_extraction import text
stop = text.ENGLISH_STOP_WORDS
import tensorflow_hub as hub
import tensorflow_text  # this needs to be imported to set up some stuff in the background
from ast import literal_eval
from sklearn.svm import SVC

def init_model(embed_model):
    # load embedding model
    embed = hub.load(embed_model)
    return embed

# print unique TN_ID and Company name
def printTNID_COName(subset_train_df):
    print(subset_train_df.shape)
    print(subset_train_df['tn_id'].value_counts())
    #print(gd_df['employer_name'].unique())
    print(subset_train_df['employer_name'].value_counts())

# Data cleaning 
def clean_text(text):
    # text = re.sub(r"\'s", " ", text)
    text = re.sub(r"don\'t", "do not ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    #text = re.sub('idk', 'i do not know', text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

# clean punctuniations
# remove non english chars 
# remove stopi words
#cols: summary, pros_description, cons_description, advice_to_management
def preProcess(gd_df, cols):
    tmp_df=gd_df
    for col_name in cols:
        print("Cleaning:", col_name)
        tmp_df[col_name]= tmp_df[col_name].str.lower()
        tmp_df[col_name]=tmp_df[col_name].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
        tmp_df[col_name]= tmp_df[col_name].apply(clean_text)
        tmp_df[col_name].replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)   
        tmp_df[col_name] = tmp_df[col_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        tmp_df[col_name]= tmp_df[col_name].str.strip()
    print(tmp_df.shape)
    return tmp_df


def process_GD_CSV(gd_csvfile): 
    gd_df=pd.read_csv(gd_csvfile)
    printTNID_COName(gd_df)
    gd_df_sub=pd.DataFrame(gd_df[['id','tn_id','as_of_date','summary','pros_description','cons_description','advice_to_management']])
    gd_df_sub.dropna(subset=['tn_id'], inplace=True)
    print("\nCheck data shape, info, number of nan\n") 
    print(gd_df_sub.shape)
    print(gd_df_sub.info())
    print(gd_df_sub.isna().sum())
    print("\nFill nan to empty_string\n")
    gd_df_sub.fillna('[SEP]', inplace=True)
    print(gd_df_sub.isna().sum())
    df_cleaned=preProcess(gd_df_sub, feat_cols)
    return df_cleaned

def embedding(df_co, feat_cols):
    df_embd=pd.DataFrame()
    for col in feat_cols:
        X=df_co[col]
        X_embedded=embed(X)
        col_embd=col+"_embedded"
        df_embd[col_embd]=np.array(X_embedded).tolist()
    #print(df_co.columns)
    #print(df_embd.columns)
    #print(df_embd.shape)
    return df_embd 

def embed_all(df_cleaned, feat_cols):
    # for each tn_id
    df_embedded_all=pd.DataFrame()
    for id in df_cleaned['tn_id'].unique():
        print("\n Processing Embedding tn_id:", id)
        df_co=df_cleaned.loc[df_cleaned['tn_id']==id]
        df_co.reset_index(drop=True, inplace=True)
        df_embd=embedding(df_co, feat_cols)
        df_tmp=pd.concat([df_co,df_embd], axis=1) #, ignore_index=True)
        df_embedded_all=pd.concat([df_embedded_all, df_tmp], axis=0, ignore_index=True)

    print("df embedded shape: ",  df_embedded_all.shape)
    return df_embedded_all

def topic_to_classid(topic_file):
    df_topics=pd.read_excel(topic_file, index_col=None)
    print(df_topics.shape)
    # to lower case
    df_topics.columns= df_topics.columns.str.lower()
    for c in df_topics.columns:
        df_topics[c]=df_topics[c].str.lower()
    print(df_topics.columns)
    # map topic name to ID
    class_id_map={}
    i=0
    lst=[]
    id=[]
    for c in df_topics.columns:
        tdf=df_topics[c].dropna()
        tmplst=tdf.to_list()
        lst=lst + tmplst
        id=id+ [i]*len(tmplst)
        class_id_map[i]=c
        i+=1
    df_topic_id=pd.DataFrame({"topic_words":lst, "class_id":id})
    return df_topics, df_topic_id

def topic_to_embedded(topic_file, embed):
    df_topics, df_topic_id=topic_to_classid(topic_file)
    X_train=df_topic_id['topic_words']
    X_train_embedded=embed(X_train)
    y_train=df_topic_id['class_id']
    print(X_train_embedded.shape)
    return X_train_embedded, y_train, df_topics, df_topic_id

def init_classifier(X_train_embedded, y_train):
    # simple SVM model for classification, using probability to output prob of each class
    svm_embedded = SVC(probability=True)
    svm_embedded.fit(X_train_embedded, y_train)
    return svm_embedded

def cls_predict(cls, df_embedded, df_topics, feat_cols, output_dir):
    probs=[]
    for col in feat_cols:
        col_embd=col+"_embedded"
        print(col_embd)
        # X_test=df_embedded[col]
        X_test_embedded=np.array(df_embedded[col_embd].to_list())
        #X_test_embedded=np.array(df_embedded[col_embd].apply(literal_eval).to_list())
        X_test_embedded=X_test_embedded.astype(float)
        y_preds_proba=cls.predict_proba(X_test_embedded)

        probs.append(y_preds_proba)
        # save probs to csv
        df_class_prob=pd.DataFrame(y_preds_proba, columns=df_topics.columns)
        #print(df_class_prob.shape)
        tmp_df=df_embedded[['id','tn_id','as_of_date',col]]
        df_out=pd.concat([df_class_prob,tmp_df], axis=1)
        outfilename=output_dir+col+"_probs.csv"
        df_out.to_csv(outfilename, index=False)
    return probs


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-m', '--model', action="store", dest="model", help="Sentence Embedding Model Dir", default=glb.embed_model)
    parser.add_option('-c', '--csv', action="store", dest="csv", help="GlassDoor Reviews csv", default=glb.gd_csvfile)
    parser.add_option('-t', '--topic', action="store", dest="topic_file", help="Topic Seedword, excel file", default=glb.topic_file) 
    parser.add_option('-o', '--outputdir', action="store", dest="outputdir", help="Output dir", default=glb.output_dir)
    options, args = parser.parse_args()

    start_time=datetime.now()
    print('Input GlassDoor Review CSV', options.csv)
    print('Input Sentence Embedding Model Dir', options.model)
    print('Input Topic Seedwords, excel file', options.topic_file)
    print('Output dir', options.outputdir)

    
    ## Step 1:  Preprocessing
    print("Preprocessing file: ", options.csv)
    print("Start Preprocessing:", datetime.now() )
    df_cleaned=process_GD_CSV(options.csv)
    print("End Preprocessing:", datetime.now() )
   
    ## Step 2: Embedding reviews
    # run sentence embedding
    print("\nStart Embedding:", datetime.now() )
    embed=init_model(options.model)
    df_embedded=embed_all(df_cleaned, feat_cols)
    print("End Embedding:", datetime.now() )

    ### DO NOT SAVE EMBEDDED REVIEWS, FILE IS TOO BIG ####
    # save embedded to csv, this file is very big 
    # output_embedded_csv=options.outputdir+"/"+"embedded.csv"
    # df_embedded.to_csv(output_embedded_csv)
    # print("\nSave Embedded to csv:", output_embedded_csv )

    ## Step 3 Load topic seedwords excel file and run embedding
    print("Build Topic : ", options.topic_file)
    X_train_embedded,y_train, df_topics, df_topic_id=topic_to_embedded(options.topic_file, embed)
    print("Finish Build Topic")


    ## Step 4 classification 
    # classifier
    print("Run Classification : ", datetime.now())
    cls = init_classifier(X_train_embedded, y_train)
    if not os.path.exists(options.outputdir):
      os.makedirs(options.outputdir)
   
    # run classification
    probs=cls_predict(cls, df_embedded, df_topics, feat_cols, options.outputdir)
    print("Finish Classification" , datetime.now())
    print("Output probs csv files at" , options.outputdir)

    end_time=datetime.now()
    print("Start_time:", start_time, " End_time", end_time)
