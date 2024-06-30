import pandas as pd
import matplotlib.pyplot as plt
from utility import get_config, read_text_to_dataframe

def plot_sum_wordcount_by_docid(df):
    # group `docid`, sum `wdcount`
    grouped_df = df.groupby('docid')['wdcount'].sum().reset_index()

    plt.figure(figsize=(14, 8))
    plt.bar(grouped_df['docid'], grouped_df['wdcount'], color='skyblue')
    plt.xlabel('docid')
    plt.ylabel('Total Word Count')
    plt.title('Total Word Count per docid')
    plt.xticks(rotation=90) 
    plt.tight_layout() 
    
    plt.show()

if __name__ == "__main__":
    df = read_text_to_dataframe(get_config('file_input'))
    #grouped_df = df.groupby('docid').sum()
    #print(grouped_df.head())
    plot_sum_wordcount_by_docid(df)