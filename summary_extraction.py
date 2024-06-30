import pandas as pd
import numpy as np
from utility import get_config, read_text_to_dataframe_merge, split_into_sentences, remove_stopwords
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# summary text
def summary_text(data, number_of_sentences):
    # split the the text in the articles into sentences
    sentences = []
    for s in data:
        sentences.append(sent_tokenize(s))

    # flatten the list
    sentences = [y for x in sentences for y in x]

    # text preprocessing
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    # removing stopwords
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    # vector representation of sentences
    word_embeddings = {}
    f = open(get_config('glove_data'), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    # create vectors for our sentences
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    # similarity matrix preparation
    sim_mat = np.zeros([len(sentences), len(sentences)])

    # use Cosine Similarity to compute the similarity between a pair of sentences.
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

    # applying PageRank algorithm
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    
    # ranking sentences
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    # extract top number_of_sentences as the summary
    print(f'summary text with number of sentences {number_of_sentences}:')
    for i in range(number_of_sentences):
        print(ranked_sentences[i][1])

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
    summary_text(sentences, number_of_sentences)
    #summary_text(sample, number_of_sentences)
