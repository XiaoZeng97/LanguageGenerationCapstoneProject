
import nltk
from sklearn import feature_extraction
import re
from nltk.tag import pos_tag
import argparse
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS



nltk.download('stopwords')
nltk.download('punkt')

stemmer = SnowballStemmer("english")
stopwords = nltk.corpus.stopwords.words('english')


def parse_csv(filtered_comments, column_name):
    row_count = filtered_comments.shape[0]
    titles = []
    synopses = []
    cluster_dict = {}
    for i in range(row_count):
        name = filtered_comments.loc[i][column_name]
        value = filtered_comments.loc[i]['body']
        value = re.sub(r'^[^:]+://([^.]+\.)+[^/]+/([^/]+/)+[^#]+(#.+)?$', "", value)

        if name not in cluster_dict:
            cluster_dict[name] = []
        cluster_dict[name].append(value)
    for k, v in cluster_dict.items():
        titles.append(k)
        synopses.append('\n'.join(v))
    filtered_comments[column_name + '_cluster'] = filtered_comments[column_name]

    return titles, synopses


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


def strip_proppers_POS(text):
    tagged = pos_tag(text.split())  # use NLTK's part of speech tagger
    non_propernouns = [word for word, pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns


def cluster_main(filtered_comments, column_name, num_clusters, min_df, max_df):

    column_cluster_name = column_name + '_cluster'
    titles, synopses = parse_csv(filtered_comments, column_name)

    ranks = []

    for i in range(0, len(titles)):
        ranks.append(i)

    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in synopses:
        allwords_stemmed = tokenize_and_stem(i)
        totalvocab_stemmed.extend(allwords_stemmed)

        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)


    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)

    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, max_features=2000000,
                                       min_df=min_df, stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

    # get_ipython().run_line_magic('time', 'tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)')

    # print(tfidf_matrix.shape)
    tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)

    terms = tfidf_vectorizer.get_feature_names()

    dist = 1 - cosine_similarity(tfidf_matrix)


    km = KMeans(n_clusters=num_clusters)

    km.fit(tfidf_matrix)


    clusters = km.labels_.tolist()




    films = {'title': titles, 'rank': ranks, 'synopsis': synopses, 'cluster': clusters}

    frame = pd.DataFrame(films, index=[clusters], columns=['rank', 'title', 'cluster'])

    # In[346]:


    frame['cluster'].value_counts()



    cluster_names = {}
    name_to_cluster_ind = {}
    output_file_name = "cluster_result_" + column_name + ".txt"
    print("Top terms per cluster:")
    print()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    with open(output_file_name, "w") as f:
        for i in range(num_clusters):
            print("Cluster %d words:" % i, end='')
            f.write("Cluster %d words:" % i)
            label_name = ""
            for ind in order_centroids[i, :20]:
                print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')
                f.write(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0])
                label_name += vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0] + ","
            cluster_names[i] = label_name
            f.write("\n\n")
            print()
            print()
            print("Cluster %d titles:" % i, end='')
            f.write("Cluster {} titles (count: {}):".format(i, len(frame.ix[i]['title'].values.tolist())))
            for title in frame.ix[i]['title'].values.tolist():
                print(' %s,' % title, end='')
                f.write(' %s,' % title)
                name_to_cluster_ind[title] = i
            print()
            f.write("\n\n")
            print()



    # This is purely to help export tables to html and to correct for my 0 start rank (so that Godfather is 1, not 0)
    frame['Rank'] = frame['rank'] + 1
    frame['Title'] = frame['title']


    row_count = filtered_comments.shape[0]

    for i in range(row_count):
        name = filtered_comments.loc[i][column_name]
        filtered_comments.loc[i, column_cluster_name] = name_to_cluster_ind[name]


    MDS()

    # two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    xs, ys = pos[:, 0], pos[:, 1]

    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

    # group by cluster
    groups = df.groupby('label')

    # set up plot
    fig, ax = plt.subplots(figsize=(30, 20))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    # iterate through groups to layer the plot
    # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x',  # changes apply to the x-axis
                       which='both',  # both major and minor ticks are affected
                       bottom='off',  # ticks along the bottom edge are off
                       top='off',  # ticks along the top edge are off
                       labelbottom='off')
        ax.tick_params(axis='y',  # changes apply to the y-axis
                       which='both',  # both major and minor ticks are affected
                       left='off',  # ticks along the bottom edge are off
                       top='off',  # ticks along the top edge are off
                       labelleft='off')

    ax.legend(numpoints=1)  # show legend with only 1 point

    # add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

    # plt.show()
    # show the plot

    # uncomment the below to save the plot if need be
    fig.savefig('clusters_visualization_' + column_name + ".png")

    plt.close()

parser = argparse.ArgumentParser()

parser.add_argument('-input_csv', type=str, required=True, help='input csv file')

parser.add_argument('-output_csv', type=str, default="result.csv", help='output csv file name')

parser.add_argument('-author_min_df', type=float, default=0.2, help='min df for author')
parser.add_argument('-author_max_df', type=float, default=0.8, help='max df for author')

parser.add_argument('-subreddit_min_df', type=float, default=0.2, help='min df for subreddit')
parser.add_argument('-subreddit_max_df', type=float, default=0.8, help='min df for subreddit')

parser.add_argument('-author_num_clusters', type=int, default=10, help='number of clusters for author')
parser.add_argument('-subreddit_num_clusters', type=int, default=8, help='number of clusters for subreddit')


args = parser.parse_args()

filtered_comments = pd.read_csv(args.input_csv,low_memory=False)
cluster_main(filtered_comments, "subreddit", args.subreddit_num_clusters, args.subreddit_min_df, args.subreddit_max_df)
cluster_main(filtered_comments, "author", args.author_num_clusters, args.author_min_df, args.author_max_df)
filtered_comments.to_csv(args.output_csv)
