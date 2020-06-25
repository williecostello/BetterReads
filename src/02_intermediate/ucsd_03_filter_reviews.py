import pandas as pd
import os
from shutil import rmtree


# Set file directories
read_dir = 'data/01_raw/ucsd_all/'
write_dir = 'data/02_intermediate/'
book_list_dir = write_dir
scratch_dir = f'{write_dir}ucsd_reviews_scratch/'

# If output directories do not exist, create them
if not os.path.isdir(write_dir):
    os.makedirs(write_dir)
if not os.path.isdir(scratch_dir):
    os.makedirs(scratch_dir)


################################################################################
# Filter each review file to only those books in list
################################################################################

# Create list of books IDs
book_list = pd.read_csv(f'{book_list_dir}ucsd_book_ids.csv')
book_list = list(book_list['book_id'])
total_reviews = 0

# List review files to be read in, ignoring hidden files
file_list = [f for f in os.listdir(read_dir) if f.startswith('reviews')]

# Loop through each file in file list
for file in file_list:

    # Read in JSON file as dataframe
    df = pd.read_json(f'{read_dir}{file}', lines=True)
    full_length = len(df)

    # Retain only the needed columns
    df = df[['book_id', 'user_id', 'rating', 'review_text', 'date_added']]

    # Rename columns to match other dataframes
    df.rename(columns={'user_id': 'reviewer_id', 'review_text': 'review',
                       'date_added': 'date'}, inplace=True)

    # Filter dataframe to only those books in list
    df = df[df['book_id'].isin(book_list)]
    trunc_length = len(df)
    total_reviews += trunc_length

    print(f'Found {trunc_length} reviews in {file} file, '
          f'dropping {full_length - trunc_length}.\n'
          f'Found {total_reviews} reviews in total.\n')

    # Write filtered dataframe to new file
    df.to_csv(f'{scratch_dir}{file}.csv', index=False)

    # Delete dataframe, to conserve memory
    del df


################################################################################
# Stitch truncated review files together
################################################################################

# List filtered review files to be stitched together
file_list = [f for f in os.listdir(scratch_dir) if f.startswith('reviews')]

# Initialize empty dataframe
df = pd.DataFrame()

# Loop through each file in file list
for file in file_list:

    # Read in CSV as dataframe & append to master dataframe
    df_file = pd.read_csv(f'{scratch_dir}{file}')
    df = df.append(df_file)

# Merge with book info dataframe to add book title & author
book_info_df = pd.read_csv(f'{write_dir}ucsd_master.csv')
df = df.join(book_info_df.set_index('book_id'), on='book_id')
df = df[['book_id', 'book_title', 'book_author', 'rating', 'review',
         'reviewer_id', 'date']]
del book_info_df

# Sort dataframe by book ID
df.sort_values(by='book_id', inplace=True)

# Write stitched dataframe to new file
df.to_csv(f'{write_dir}ucsd_reviews.csv', index=False)

# Remove scratch directory
rmtree(scratch_dir)