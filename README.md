# BetterReads

An interactive app that rapidly extracts the most commonly expressed opinions across all of a book's reviews.

## Introduction

BetterReads is a Python-based app that uses natural language processing and unsupervised machine learning to distill thousands of reviews of a particular book down to their most commonly expressed opinions. It accomplishes this by dividing each review into its individual sentences (using NLTK), transforming each sentence into a 512-dimensional vector (using the [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4)), finding the densest regions in the resultant vector space (using k-means clustering), and extracting the sentences that are closest to the centre of each region. To learn more...

- **Visit the app:** [bit.ly/betterreads](https://bit.ly/betterreads)
- **Read the blog post:** [bit.ly/whatisbetterreads](https://bit.ly/whatisbetterreads)
- **Watch a video walkthrough:** [youtu.be/ojkM4JGT93k](https://youtu.be/ojkM4JGT93k)

## Notebooks

Detailed explanations of how the algorithm was built and the exploratory analysis involved are presented in the included Jupyter notebooks.

- `01_modelling_with_use` Establishes proof-of-concept of BetterReads algorithm
- `02_optimizing_kmeans` Explores how to optimize k-means clustering model
- `03_optimizing_reviews` Explores how many reviews are needed to obtain meaningful results
- `04_optimizing_goodreads` Explores how to tune model on GoodReadsReviewsScraper data
- `05_optimizing_use` Compares performance between versions of Universal Sentence Encoder
- `06_visualizing_results` Creates some simple visualizations of our model

## How to run

After cloning the repo and creating a virtual environment with the packages listed in requirements.txt, run the setup.sh script to create the expected file directories and install the expected data. **Note that not all datasets can be downloaded with this script**, so be sure to follow the instructions in the script's comments.

Scripts to clean, process, and transform the data are included under `src`. **All scripts should be run from the root directory**, to preserve file directory references.

The web app script can be run locally with the following command:

```
streamlit run src/05_web_app/betterreads.py
```

Note that the web app expects to find sentence datasets in `src/05_web_app/data` and book cover images in `src/05_web_app/book_covers`, and downloads sentence embeddings from a private S3 bucket. The sentence datasets are identical to the files written to `data/03_processed` and the sentence vectors are identical to the files written to `data/05_model_output`. The app's code can easy be rewritten to load these files locally, and the book cover images can be commented out.

## Data sources

Data for the BetterReads algorithm comes from two sources:

- [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/reviews?authuser=0): full text of 15.7 million GoodReads reviews, of 2M books from 465K users, scraped in late 2017
- [GoodReadsReviewsScraper](https://github.com/williecostello/GoodReadsReviewsScraper): self-made web-scraping script that, from any GoodReads book URL (or list of URLs), downloads the full text of its top 1,500 reviews

Instructions for downloading the UCSD Book Graph data are included in `setup.sh`. Data generated with the scraper should copied to `data/01_raw/goodreads/`.

## Directory structure

```
├── setup.sh            <- Script to download data files
│
├── requirements.txt    <- Required Python packages
│
├── data                <- Data directory (created in setup.sh)
│   ├── 01_raw              <- Raw data from UCSD Book Graph / GoodReadReviewsScraper
│   ├── 02_intermediate     <- Cleaned UCSD book review datasets
│   ├── 03_processed        <- Processed book review datasets
│   ├── 04_models           <- Universal Sentence Encoder (installed in setup.sh)
│   └── 05_model_output     <- Sentence embeddings
│
├── notebooks           <- Jupyter notebooks detailing analysis
│
└── src                 <- All scripts to be run from root directory
    ├── 01_data             <- Splits UCSD data into smaller chunks
    │   └── split_ucsd_files.sh
    │
    ├── 02_intermediate     <- Cleans UCSD data
    │   ├── ucsd_01_compile_book_list.py
    │   ├── ucsd_02_compile_master_list.py
    │   ├── ucsd_03_filter_reviews.py
    │   └── ucsd_04_write_book_files.py
    │
    ├── 03_processing       <- Processes review datasets into sentence datasets
    │   ├── process_reviews.py
    │   └── process_utils.py
    │
    ├── 04_modelling        <- Embeds sentence datasets as vectors
    │   └── embed_sentences.py
    │
    └── 05_web_app          <- Runs Streamlit web app (bit.ly/betterreads) 
	     └── betterreads.py
```

## Thanks

Special thanks to [Mengting Wan](https://mengtingwan.github.io/) and [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/) of UCSD, for making the Book Graph dataset freely available online; see their papers ["Item Recommendation on Monotonic Behavior Chains"](https://mengtingwan.github.io/paper/recsys18_mwan.pdf) and ["Fine-Grained Spoiler Detection from Large-Scale Review Corpora"](https://www.aclweb.org/anthology/P19-1248/).  

[Kushal Chauhan](https://medium.com/@kushalchauhan)'s blog post on ["Unsupervised Text Summarization using Sentence Embeddings"](https://medium.com/jatana/unsupervised-text-summarization-using-sentence-embeddings-adb15ce83db1) was a huge help and a big inspiration early on.

The BetterReads app was made as part of my capstone for the Data Science Diploma Program at [BrainStation](https://brainstation.io). Many many thanks to my educators: [Myles Harrison](https://www.linkedin.com/in/mylesharrison/), [Adam Thorstenstein](https://www.linkedin.com/in/adamjthor/?originalSubdomain=ca), [Govind Suresh](https://www.linkedin.com/in/govindsuresh/), [Patrick Min](https://www.linkedin.com/in/pmin/), and [Daria Aza](https://www.linkedin.com/in/dariaaza/). Thanks also to all my fantastic classmates!