import configparser
import os
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import spacy
nlp = spacy.load('en_core_web_sm')

def get_config(key):
    config = configparser.ConfigParser()
    config.read(os.path.dirname(__file__) + '/config.ini')

    if 'config' not in config:
        raise Exception("Section 'config' not found in config.ini.")
        
    return config.get('config', key)

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)

def read_text_to_dataframe_merge(file_path):
    df = read_text_to_dataframe(file_path)
    merged_df = df.groupby('docid')['text'].apply(' '.join).reset_index()

    return merged_df

def read_text_to_dataframe(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    soup = BeautifulSoup(content, 'html.parser')

    data = []
    for s_tag in soup.find_all('s'):
        docid = s_tag.get('docid')
        num = int(s_tag.get('num'))
        wdcount = int(s_tag.get('wdcount'))
        text = s_tag.get_text()
        
        data.append({'docid': docid, 'num': num, 'wdcount': wdcount, 'text': text})
    
    return pd.DataFrame(data)

# function to remove stopwords
def remove_stopwords(sen):
    #import the stopwords.
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]