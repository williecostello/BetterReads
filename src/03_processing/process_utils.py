import pandas as pd
from langdetect import detect
from nltk.tokenize import sent_tokenize

# Function to clean review text
def clean_reviews(file_name, file_path, goodreads=False):

    # Read in CSV as dataframe
    df = pd.read_csv(f'{file_path}{file_name}')
    length_orig = len(df)

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    num_dups = length_orig - len(df)

    print(f'Read in {length_orig} reviews, dropping {num_dups} duplicates\n')

    # Define spoiler marker & "...more" strings, and remove from all reviews
    spoiler_str_ucsd = '\*\* spoiler alert \*\* \n'
    spoiler_str_gr = '                    This review has been hidden because it contains spoilers. To view it,\n                    click here.\n\n\n'
    more_str = '\n...more\n\n'
    df['review'] = df['review'].str.replace(spoiler_str_ucsd, '')
    df['review'] = df['review'].str.replace(spoiler_str_gr, '')
    df['review'] = df['review'].str.replace(more_str, '')

    # Scraped reviews from GoodReads typically repeat the first ~500 characters
    # The following loop removes these repeated characters
    if goodreads:

        # Create unique review index for each review
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'review_index'}, inplace=True)

        # Loop through each row in dataframe
        for i in range(len(df)):

            # Save review and review's first ~250 characters to variables
            review = df.iloc[i]['review']
            review_start = review[2:250]

            # Loop through all of review's subsequent character strings
            for j in range(3, len(review)):

                # Check if string starts with same sequence as review start
                if review[j:].startswith(review_start):
                    # If so, chop off all previous characters from review
                    df.at[i, 'review'] = review[j:]

    # Replace all new line characters
    df['review'] = df['review'].str.replace('\n', ' ')

    # Append space to all sentence end characters
    df['review'] = df['review'].str.replace('.', '. ').replace('!', '! ').replace('?', '? ')

    # Initialize dataframe to store English-language reviews
    reviews_df = pd.DataFrame()
    # Initialize counter for dropped reviews
    drop_ctr = 0

    # Loop through each row in dataframe
    for i in range(len(df)):

        # Save review to variable
        review = df.iloc[i]['review']

        # Check if review is English
        try:
            if detect(review) == 'en':
                # If so, add row to English-language dataframe
                reviews_df = reviews_df.append(df.loc[i, :])
            else:
                # If not, add 1 to dropped review counter
                drop_ctr += 1
        # If check fails, add 1 to dropped review counter
        except:
            drop_ctr += 1

    print(f'Dropped {drop_ctr} non-English reviews. '
          f'{len(reviews_df)} reviews remain.\n')

    return reviews_df


# Function to tokenize review text into sentences
def make_sentences(reviews_df):

    # Initialize dataframe to store review sentences, and counter
    sentences_df = pd.DataFrame()
    ctr = 0

    print(f'Starting tokenization')

    # Loop through each review
    for i in range(len(reviews_df)):

        # Save row and review to variables
        row = reviews_df.iloc[i]
        review = row.loc['review']

        # Tokenize review into sentences
        sentences = sent_tokenize(review)

        # Loop through each sentence in list of tokenized sentences
        for sentence in sentences:
            # Add row for sentence to sentences dataframe
            new_row = row.copy()
            new_row.at['review'] = sentence
            sentences_df = sentences_df.append(new_row, ignore_index=True)

        ctr += 1
        if (ctr % 500 == 0):
            print(f'{ctr} reviews tokenized')

    print(f'Tokenization complete: {len(sentences_df)} sentences tokenized\n')

    # Rename review column
    sentences_df.rename(columns={'review':'sentence'}, inplace=True)

    return sentences_df


# Function to clean sentences
def clean_sentences(sentences_df, lower_thresh=5, upper_thresh=50):

    # Remove whitespaces at the start and end of sentences
    sentences_df['sentence'] = sentences_df['sentence'].str.strip()

    # Create list of sentence lengths
    sentence_lengths = sentences_df['sentence'].str.split(' ').map(len)

    num_short = (sentence_lengths <= lower_thresh).sum()
    num_long = (sentence_lengths >= upper_thresh).sum()
    num_sents = num_short + num_long

    # Filter sentences
    sentences_df = sentences_df[
        (sentence_lengths > lower_thresh) & (sentence_lengths < upper_thresh)]

    print(f'Dropped {num_sents} sentences: {num_short} short, {num_long} long\n'
          f'Final dataset contains {len(sentences_df)} sentences.\n')

    return sentences_df