# Web app packages
import streamlit as st
import boto3
import joblib

# Data science packages
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from random import shuffle


# Sidebar panel
st.sidebar.markdown(
    '''
    # BetterReads
    *Curated crowdsourcing of book reviews*
    
    ---
    '''
)
book = st.sidebar.selectbox(
    'Select a book:', ('',
        'Cloud Atlas',
        'Gone Girl',
        'Lincoln in the Bardo',
        "Mr. Penumbra's 24-Hour Bookstore",
        'My Brilliant Friend',
        'Never Let Me Go',
        'Normal People',
        'Neverwhere',
        'Ready Player One',
        'Recursion',
        'Station Eleven',
        'The Brief Wondrous Life of Oscar Wao',
        'The Fault in Our Stars',
        'The Fifth Season',
        'The Girl on the Train',
        'The Kite Runner',
        'The Luminaries',
        'The Martian',
        'The Name of the Wind',
        'The Overstory',
        'The Silent Patient',
        'The Testaments',
        'The Three-Body Problem',
        'This Is How You Lose the Time War',
        'World War Z'
    )
)
k = st.sidebar.slider(
    'Select number of distinct opinions to find in reviews:',
    3, 9, value=6, step=1
)
n = st.sidebar.slider(
    'Select number of examples to show for each opinion:',
    1, 8, value=3, step=1
)
stars = st.sidebar.slider(
    'Filter reviews by star rating:',
    1, 5, value=(1, 5), step=1
)
st.sidebar.markdown(
    '''
    ---
    an original app by [Willie Costello](https://williecostello.com)
    '''
)


file_dict = {
    'The Fault in Our Stars':'11870085_the_fault_in_our_stars',
    "Mr. Penumbra's 24-Hour Bookstore":"13538873_mr._penumbra's_24-hour_bookstore_(mr._penumbra's_24-hour_bookstore,_#1)",
    'My Brilliant Friend':'13586707_my_brilliant_friend_(the_neapolitan_novels_#1)',
    'Neverwhere':'14497_neverwhere',
    'The Luminaries':'17333230_the_luminaries',
    'The Martian':'18007564_the_martian',
    'The Name of the Wind':'186074_the_name_of_the_wind_(the_kingkiller_chronicle,_#1)',
    'The Fifth Season':'19161852_the_fifth_season_(the_broken_earth,_#1)',
    'Station Eleven':'20170404_station_eleven',
    'The Three-Body Problem':'20518872_the_three-body_problem_(remembrance_of_earth’s_past,_#1)',
    'The Girl on the Train':'22557272_the_girl_on_the_train',
    'The Brief Wondrous Life of Oscar Wao':'297673_the_brief_wondrous_life_of_oscar_wao',
    'Lincoln in the Bardo':'29906980_lincoln_in_the_bardo',
    'The Silent Patient':'40097951_the_silent_patient',
    'The Overstory':'40180098_the_overstory',
    'Normal People':'41057294_normal_people',
    'Recursion':'42046112_recursion',
    'The Testaments':'42975172_the_testaments',
    'This Is How You Lose the Time War':'43352954_this_is_how_you_lose_the_time_war',
    'Cloud Atlas':'49628_cloud_atlas',
    'Never Let Me Go':'6334_never_let_me_go',
    'The Kite Runner':'77203_the_kite_runner',
    'Gone Girl':'8442457_gone_girl',
    'World War Z':'8908_world_war_z_an_oral_history_of_the_zombie_war',
    'Ready Player One':'9969571_ready_player_one'}


# Initialize S3 client for loading data
s3_client = boto3.client('s3')


@st.cache
def load_reviews(csv_file, pkl_file):
    '''
    Load sentences & sentence vectors from betterreads S3 bucket 
    '''
    # Download pickle file from S3
    s3_client.download_file('betterreads', pkl_file, 'vectors.pkl')

    # Read in sentences CSV as dataframe & filter by selected star ratings
    df = pd.read_csv(f'data/{csv_file}')
    filter_index = df[(df['rating'] >= stars[0]) & (df['rating'] <= stars[1])].index
    df = df.loc[filter_index].reset_index(drop=True)
    sentences = df['sentence'].copy()

    # Read in sentence vectors & filter similarly
    sentence_vectors = joblib.load('vectors.pkl')
    sentence_vectors = sentence_vectors[filter_index]

    return sentences, sentence_vectors


@st.cache
def find_opinions(sentence_vectors, k):
    '''
    Run a k-means model on sentence vectors & return cluster centres
    '''
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(sentence_vectors)
    cluster_centres = kmeans_model.cluster_centers_
    return cluster_centres


@st.cache
def get_opinions(sentences, sentence_vectors, cluster_centres, k):
    '''
    Find the ten closest sentences to each cluster centre
    '''
    # Initialize dataframe to store cluster centre sentences
    df = pd.DataFrame()
    
    # Set the number of cluster centre points to look at when calculating uniformity score
    centre_points = int(len(sentences) * 0.01)
    
    # Loop through number of clusters selected
    for i in range(k):

        # Define cluster centre
        centre = cluster_centres[i]

        # Calculate inner product of cluster centre and sentence vectors
        ips = np.inner(centre, sentence_vectors)

        # Find the 10 sentences with the highest inner products
        top_indices = pd.Series(ips).nlargest(10).index
        top_sentences = list(sentences[top_indices])

        # Randomly shuffle top sentences (for variety)
        # shuffle(top_sentences)
        
        # Calculate uniformity score for cluster
        centre_ips = pd.Series(ips).nlargest(centre_points)
        uniformity_score = np.mean(centre_ips)
        
        # Create new row with cluster's top 10 sentences and uniformity score
        new_row = pd.Series([top_sentences, uniformity_score])
        
        # Append new row to master dataframe
        df = df.append(new_row, ignore_index=True)
        
    # Rename dataframe columns
    df.columns = ['sentences', 'Uniformity']

    # Sort dataframe by uniformity score, from highest to lowest
    df = df.sort_values(by='Uniformity', ascending=False).reset_index(drop=True)

    return df


if book != '':
    '''
    # What are people saying about...
    ###
    '''

    book_id = file_dict[book].split('_')[0]
    st.image(f'book_covers/{book_id}.jpg')

    '---'

    # Create S3 file names
    csv_file = f'{file_dict[book]}.csv'
    pkl_file = f'{file_dict[book]}.pkl'

    # Load sentences & sentence vectors
    sentences, sentence_vectors = load_reviews(csv_file, pkl_file)

    # Find cluster centres
    cluster_centres = find_opinions(sentence_vectors, k)

    # Find the ten closest sentences to each cluster centre
    sentences_df = get_opinions(sentences, sentence_vectors, cluster_centres, k)

    uni_scores = dict()

    # Loop through number of clusters selected
    for i in range(k):
        
        # Save uniformity score & sentence list to variables
        uni_score = round(sentences_df.loc[i]['Uniformity'], 3)
        uni_scores.update({f'Opinion #{i+1}':uni_score})
        sents = sentences_df.loc[i]['sentences'].copy()
        
        f'**Opinion #{i+1}**'

        # Print out number of sentences selected
        for j in range(n):
            f'- {sents[j]}'

        '---'
    
    '''
    ### Output analysis: Opinion uniformity scores
    '''
    uniformity_df = pd.DataFrame()
    uniformity_df = uniformity_df.append(uni_scores, ignore_index=True)
    uniformity_df.index = ['']
    st.table(uniformity_df)
    '''
    *Each opinion's uniformity score reflects the amount of semantic uniformity across that opinion's most central sentences. Generally speaking, the higher an opinion's uniformity score, the more confident we can be that its sentences represent a widespread opinion within the full set of reviews.*
    
    *Scores range between 0 and 1, though any value above 0.4 should be considered high. Uniformity scores will change as the number of opinions is adjusted. Opinions are automatically displayed in descending order of their uniformity scores.* 
    '''


if book == '':
    '''
    # :books: BetterReads
    
    *Rapidly extract the most commonly expressed opinions across all of a book's reviews*


    ### :raising_hand: What to do
    
    Select a book from the dropdown menu on the left. The BetterReads algorithm will then search that book's reviews, find the most commonly expressed opinions, and display them all to you.
    
    You may also adjust the number of opinions to find and the number of example sentences for each opinion to display, and filter the reviews by their star rating.
    
    *Don't worry if it seems to be taking a while! That just means the algorithm is working extra hard* :wink:


    ### :book: The books
    '''

    st.image('book_covers/cover_graphic.jpg', use_column_width=True)

    '''
    ### :spiral_note_pad: The reviews

    All reviews used in this app originate from [GoodReads](https://www.goodreads.com/). Some come from the [UCSD Book Graph dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home); others were scraped with the help of my self-made [GoodReadsReviewsScraper](https://github.com/williecostello/GoodReadsReviewsScraper).


    ### :floppy_disk: How it works

    Sentence embeddings, k-means clustering, inner products... Want to learn more? Read [my tell-all blog post](https://bit.ly/whatisbetterreads)! The full code for this app and all the accompanying analysis can be found on [Github](https://github.com/williecostello/BetterReads).


    ### :pray: Thanks

    Special thanks to [Mengting Wan](https://mengtingwan.github.io/) and [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/) of UCSD, for making the Book Graph dataset freely available online; see their papers ["Item Recommendation on Monotonic Behavior Chains"](https://mengtingwan.github.io/paper/recsys18_mwan.pdf) and ["Fine-Grained Spoiler Detection from Large-Scale Review Corpora"](https://www.aclweb.org/anthology/P19-1248/).  

    [Kushal Chauhan](https://medium.com/@kushalchauhan)'s blog post on ["Unsupervised Text Summarization using Sentence Embeddings"](https://medium.com/jatana/unsupervised-text-summarization-using-sentence-embeddings-adb15ce83db1) was a huge help and a big inspiration early on.

    This app was made as part of my capstone for the Data Science Diploma Program at [BrainStation](https://brainstation.io). Many many thanks to my educators: [Myles Harrison](https://www.linkedin.com/in/mylesharrison/), [Adam Thorstenstein](https://www.linkedin.com/in/adamjthor/?originalSubdomain=ca), [Govind Suresh](https://www.linkedin.com/in/govindsuresh/), [Patrick Min](https://www.linkedin.com/in/pmin/), and [Daria Aza](https://www.linkedin.com/in/dariaaza/). Thanks also to all my fantastic classmates!
    '''