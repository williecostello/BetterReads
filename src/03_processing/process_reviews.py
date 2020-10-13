import os
from process_utils import clean_reviews, make_sentences, clean_sentences

# Set collection
goodreads = True

# Set file directories
read_dir = 'data/01_raw/goodreads/' if goodreads \
    else f'data/02_intermediate/ucsd_reviews/'
write_dir = f'data/03_processed/'

# If output directory does not exist, create it
if not os.path.isdir(write_dir):
    os.mkdir(write_dir)

# List review files to be read in, ignoring hidden files
file_list = [f for f in os.listdir(read_dir) if f.endswith('.csv')]

# Set loop variables
num_files = len(file_list)
file_index = 0

# Loop through files in file list
for file_name in file_list[file_index:]:

    file_index += 1
    print('-------------------------------------------------------------------')
    print(f'Processing file {file_index} of {num_files}: {file_name}\n')

    # Clean file's review text
    reviews_df = clean_reviews(file_name, read_dir)

    # Tokenize reviews into sentences
    all_sentences_df = make_sentences(reviews_df)

    # Clean sentences
    sentences_df = clean_sentences(all_sentences_df)
    sentences_df.to_csv(f'{write_dir}{file_name}', index=False)
