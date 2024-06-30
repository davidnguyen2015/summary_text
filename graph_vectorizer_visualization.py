from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import matplotlib.pyplot as plt
from utility import get_config, read_text_to_dataframe_merge, split_into_sentences

def graph_view(data):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), stop_words="english")
    vectorizer.fit_transform(data)

    G = nx.DiGraph()
    G.add_nodes_from(vectorizer.get_feature_names_out())

    all_edges = []
    for s in data:
        edges = []
        previous = None
        for w in s.split():
            w = w.lower()
            if w in vectorizer.get_feature_names_out():
                if previous:
                    edges.append((previous, w))
                previous = w   

        all_edges.append(edges)

    plt.figure(figsize=(20,20))
    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size = 500)
    nx.draw_networkx_labels(G, pos)
    for i, edges in enumerate(all_edges):
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='gray', arrows=True)
    plt.title("Sentence Graph Based on Shared Keywords")
    plt.show()
    plt.close("all")

if __name__ == "__main__":
    df = read_text_to_dataframe_merge(get_config('file_input'))
    sample = [
            "Trump says that it is useful to win the next presidential election",
            "The Prime Minister suggests the name of the winner of the next presidential election",
            "In yesterday conference, the Prime Minister said that it is very important to win the next presidential election",
            "The Chinese Minister is in London to discuss about climate change",
            "The president Donald Trump states that he wants to win the presidential election. This will require a strong media engagement",
            "The president Donald Trump states that he wants to win the presidential election. The UK has proposed collaboration",
            "The president Donald Trump states that he wants to win the presidential election. He has the support of his electors",
            ]
    #graph_view(sample)

    sentences = split_into_sentences(df.iloc[0]['text'])
    graph_view(sentences)