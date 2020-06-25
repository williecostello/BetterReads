import pandas as pd

# Set file directories
authors_file = 'data/01_raw/ucsd/authors.json'
books_file = 'data/02_intermediate/ucsd_books.csv'
write_dir = 'data/02_intermediate/'

# Read in authors JSON file as dataframe
authors_df = pd.read_json(authors_file, lines=True)

# Select only author ID and name columns
authors_df = authors_df.loc[:, ['author_id', 'name']]

# Rename name column
authors_df.rename(columns={'name':'book_author'}, inplace=True)

# Read in books CSV as dataframe
books_df = pd.read_csv(books_file)

# Join together books and authors dataframe on author ID column
master_df = books_df.join(authors_df.set_index('author_id'), on='author_id', how='left')

# Drop rows with missing values
master_df.dropna(inplace=True)

# Convert review count and author ID columns to integers
master_df.loc[:, ['review_count']] = master_df.loc[:, ['review_count']].astype(int)
master_df.loc[:, ['author_id']] = master_df.loc[:, ['author_id']].astype(int)

# Sort dataframe by review count, in descending order
master_df.sort_values(by='review_count', ascending=False, inplace=True)

# Write dataframe to CSV
master_df.to_csv(f'{write_dir}ucsd_master.csv', index=False)