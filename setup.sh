# Create data directories
mkdir -p data/01_raw data/02_intermediate data/03_processed data/04_models data/05_model_output


# Download Universal Sentence Encoder from TensorFlow Hub (~1 GB)
curl -L https://tfhub.dev/google/universal-sentence-encoder/4?tf-hub-format=compressed -o data/04_models/use.tar.gz
tar -xf data/04_models/use.tar.gz -C data/04_models/universal_sentence_encoder/
rm data/04_models/use.tar.gz


# UCSD Book Graph data can be downloaded from the following links
# ***Reviews and Books files are provided through Google Drive and too large to be downloaded with curl
# ***Download .gz files into root directory and execute the following commands

# Reviews: https://drive.google.com/uc?id=1pQnXa7DWLdeUpvUFsKusYzwbA5CAAZx7
# Books: https://drive.google.com/uc?id=1LXpK1UfqtP89H1tYy0pBGHjYk8IhigUK
# Authors:
curl -L https://drive.google.com/uc?id=19cdwyXwfXx_HDIgxXaHzH0mrx8nMyLvC -o goodreads_book_authors.json.gz

gunzip goodreads_reviews_dedup.json.gz
gunzip goodreads_books.json.gz
gunzip goodreads_book_authors.json.gz

mkdir data/01_raw/ucsd
mv goodreads_reviews_dedup.json data/01_raw/ucsd/reviews.json
mv goodreads_books.json data/01_raw/ucsd/books.json
mv goodreads_book_authors.json data/01_raw/ucsd/authors.json


# GoodReads data can be downloaded with the GoodReadsReviewsScraper
# https://github.com/williecostello/GoodReadsReviewsScraper
# Place all generated files in the following directory
mkdir data/01_raw/goodreads


# Download .csv data files for notebooks
curl -L https://williecostello.com/upload/goodreads_notebooks_data.zip -o goodreads_notebooks_data.zip
unzip goodreads_notebooks_data.zip
mv goodreads_notebooks_data notebooks/data
rm goodreads_notebooks_data.zip