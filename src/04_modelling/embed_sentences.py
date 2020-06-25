import numpy as np
import pandas as pd
import tensorflow_hub as hub
import os
import joblib

# Loads Universal Sentence Encoder locally, from downloaded module
embed = hub.load('data/04_models/universal_sentence_encoder/')
# Loads Universal Sentence Encoder remotely, from Tensorflow Hub
# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Set directories
read_dir = 'data/03_processed/'
write_dir = 'data/05_model_output/'

# List all CSV files in read directory
file_list = [f for f in os.listdir(read_dir) if f.endswith('.csv')]
ctr = 0

# Loop through file in file list
for file in file_list:

    # Read in processed file as dataframe
    df = pd.read_csv(f'{read_dir}{file}')
    file_stem = file.replace('.csv', '')

    # Sample dataframe down to 10000 rows if greater
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=24).reset_index(drop=True)

    # Vectorize sentences
    sentence_vectors = embed(df['sentence'])

    # Transform Tensor object to Numpy array
    sentence_array = np.array(sentence_vectors)

    # Pickle array
    joblib.dump(sentence_array, f'{write_dir}{file_stem}.pkl')

    # Create new dataframe with just sentence and rating columns
    export_df = df[['sentence', 'rating']].copy()

    # Export dataframe
    export_df.to_csv(f'{write_dir}{file_stem}.csv', index=False)

    ctr += 1

    print(f'Finished {ctr} of {len(file_list)} ({file_stem})')