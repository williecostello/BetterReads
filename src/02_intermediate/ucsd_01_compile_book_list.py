import numpy as np
import pandas as pd
import os

# Set file directories
read_dir = 'data/01_raw/ucsd_all/'
write_dir = 'data/02_intermediate/'

# List book files to be read in
file_list = [f for f in os.listdir(read_dir) if f.startswith('books')]

# Initialize master dataframe
books_df = pd.DataFrame()

# Initialize counter
counter = 0

# Loop through files in file list
for file in file_list:

    # Read in JSON file as dataframe
    df = pd.read_json(f'{read_dir}{file}', lines=True)

    # Filter dataframe to only essential columns
    df = df[['book_id', 'title', 'text_reviews_count', 'authors']]

    # Extract author IDs from authors column
    author_ids = df.loc[:, 'authors'].map(
        lambda x: int(x[0]['author_id']) if bool(x) else np.nan)

    # Create new column with author IDs
    df.loc[:, 'author_id'] = author_ids

    # Drop authors columns
    df.drop('authors', axis=1, inplace=True)

    # Append rows to master dataframe
    books_df = books_df.append(df)

    del df
    counter += 1
    print(f'Processed {counter} of {len(file_list)} files')

# Rename columns
books_df.rename(columns={'title':'book_title',
                         'text_reviews_count':'review_count'},
                inplace=True)

# Write master dataframe to CSV
books_df.to_csv(f'{write_dir}ucsd_books.csv', index=False)