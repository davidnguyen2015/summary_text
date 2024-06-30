import nltk
import spacy
import networkx as nx
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from utility import get_config, read_text_to_dataframe_merge, split_into_sentences

# ensure you have downloaded the required resources
# use command: python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

# preprocessing function
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return tokens

def graph_view(data):
    # extract keywords
    keywords = [preprocess(sentence) for sentence in data]

    # create a graph
    G = nx.Graph()

    # add nodes
    for i in range(len(data)):
        G.add_node(f"S{i+1}", sentence=data[i])

    # add edges based on shared keywords
    for i in range(len(keywords)):
        for j in range(i + 1, len(keywords)):
            if set(keywords[i]).intersection(set(keywords[j])):
                G.add_edge(f"S{i+1}", f"S{j+1}")

    # draw the graph (optional, for visualization)
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', linewidths=1, font_size=10, font_color='black', font_family='DejaVu Sans')
    plt.title("Sentence Graph Based on Shared Keywords")
    plt.show()

if __name__ == "__main__":
    df = read_text_to_dataframe_merge(get_config('file_input'))

    sentences = split_into_sentences(df.iloc[0]['text'])
    graph_view(sentences)