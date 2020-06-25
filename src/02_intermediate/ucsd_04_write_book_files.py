import pandas as pd
import os


# Set file directories
read_dir = 'data/02_intermediate/'
write_dir = f'{read_dir}ucsd_reviews/'

# If output directory does not exist, create it
if not os.path.isdir(write_dir):
    os.makedirs(write_dir)


################################################################################
# Write individual files for all books in dataframe
################################################################################

# Read in master CSV as dataframe
df = pd.read_csv(f'{read_dir}ucsd_reviews.csv')

# Create list of all book IDs in dataframe
book_ids = df['book_id'].unique()
print(f'Final dataframe contains reviews of {len(book_ids)} unique books.')

# Loop through each book ID
for book_id in book_ids:

    # Create dataframe of only that book's reviews
    book_df = df[df['book_id'] == book_id].copy()

    # Create unique review index for each review
    book_df.reset_index(drop=True, inplace=True)
    book_df.reset_index(inplace=True)
    book_df.rename(columns={'index':'review_index'}, inplace=True)

    # Create unique file name from book ID and title
    book_title = book_df.loc[0, 'book_title']
    book_title = book_title.lower().replace(':', '').replace('//', '-').replace(' ', '_')
    file_name = f'{book_id}_{book_title}'

    # Write reviews dataframe to csv
    book_df.to_csv(f'{write_dir}{file_name}.csv', index=False)