# Split UCSD Reviews and Books file into smaller chunks, to assist with data processing

mkdir data/01_raw/ucsd_all/
echo 'Created file directory'

split -l 500000 data/01_raw/ucsd/reviews.json data/01_raw/ucsd_all/reviews_
echo 'Split master reviews file'

split -l 100000 data/01_raw/ucsd/books.json data/01_raw/ucsd_all/books_
echo 'Split master books file'