import re
import nltk
from nltk.corpus import stopwords
import heapq
from utility import get_config, read_text_to_dataframe_merge, split_into_sentences

def summary_text(data, number_of_sentences):
    # fetching the data
    text = ""
    for s in data:
        text += s

    # preprocessing the data
    text = re.sub(r'',' ',text)
    text = re.sub(r'\s+',' ',text)
    clean_text = text.lower()
    clean_text = re.sub(r'\W',' ',clean_text)
    clean_text = re.sub(r'\d',' ',clean_text)
    clean_text = re.sub(r'\s+',' ',clean_text)

    sentences = nltk.sent_tokenize(text)

    # creating histogram
    word2count = {}
    stop_words = stopwords.words('english')
    for word in nltk.word_tokenize(clean_text):
        if word not in stop_words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1

    max_count = max(word2count.values())

    # weighted histogram
    word2countW = {}
    for key in word2count.keys():
        word2countW[key] = word2count[key]/max_count

    # calculating sentence score
    sent2score = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word2countW.keys():
                if len(sentence.split(' '))<25:
                    if sentence not in sent2score.keys():
                        sent2score[sentence] = word2countW[word]
                    else:
                        sent2score[sentence] += word2countW[word]

    # generating summary
    summary = heapq.nlargest(3,sent2score,key= sent2score.get)
    i=1
    for sentence in summary:
        print(f"{i}.{sentence}")
        i +=1

if __name__ == "__main__":
    df = read_text_to_dataframe_merge(get_config('file_input'))
    sentences = split_into_sentences(df.iloc[0]['text'])
    sample = [
        "Hurricane Gilbert swept toward the Dominican Republic Sunday, and the Civil Defense alerted its heavily populated south coast to prepare for high winds, heavy rains and high seas.",
        "The storm was approaching from the southeast with sustained winds of 75 mph gusting to 92 mph.",
        "``There is no need for alarm,'' Civil Defense Director Eugenio Cabral said in a television alert shortly before midnight Saturday.",
        "Cabral said residents of the province of Barahona should closely follow Gilbert's movement.",
        "An estimated 100,000 people live in the province, including 70,000 in the city of Barahona, about 125 miles west of Santo Domingo.",
        "Tropical Storm Gilbert formed in the eastern Caribbean and strengthened into a hurricane Saturday night.",
        "The National Hurricane Center in Miami reported its position at 2 a.m. Sunday at latitude 16.1 north, longitude 67.5 west, about 140 miles south of Ponce, Puerto Rico, and 200 miles southeast of Santo Domingo.",
        "The National Weather Service in San Juan, Puerto Rico, said Gilbert was moving westward at 15 mph with a ``broad area of cloudiness and heavy weather'' rotating around the center of the storm.",
        "The weather service issued a flash flood watch for Puerto Rico and the Virgin Islands until at least 6 p.m. Sunday.",
        "Strong winds associated with the Gilbert brought coastal flooding, strong southeast winds and up to 12 feet feet to Puerto Rico's south coast.",
        "There were no reports of casualties.",
        "San Juan, on the north coast, had heavy rains and gusts Saturday, but they subsided during the night.",
        "On Saturday, Hurricane Florence was downgraded to a tropical storm and its remnants pushed inland from the U.S. Gulf Coast.",
        "Residents returned home, happy to find little damage from 80 mph winds and sheets of rain.",
        "Florence, the sixth named storm of the 1988 Atlantic storm season, was the second hurricane.",
        "The first, Debby, reached minimal hurricane strength briefly before hitting the Mexican coast last month ."
        ]
    
    number_of_sentences = 3
    #summary_text(sentences, number_of_sentences)
    summary_text(sample, number_of_sentences)