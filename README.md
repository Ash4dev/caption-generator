![image](https://github.com/Ash4dev/caption-generator/assets/77205433/5d104807-c3ea-4158-9922-7b188c15a5dd)# Caption Generation using Flicker8k Dataset
The model generates captions relevant to the visual content of the images by combining image features and linguistic context. This approach utilizes encoder-decoder deep learning architectures (CNNs and LSTMs) to generate meaningful captions for images.

Hosted @: [streamlit](https://caption-generator.streamlit.app/)

Notebook @: [kaggle](https://www.kaggle.com/code/ashutoshpanigrahy/notebook0ecb3cce5b)

## Dataset Description
This repository contains code for a caption generator built using the Flicker8k dataset from Kaggle. The Flicker8k dataset consists of 8,100 images, each accompanied by 5 captions. In this project, approximately 80% (6,400 images) of the dataset was used for training, while the remaining 20% was used for testing. Flicker8k was chosen over datasets larger datasets like Flicker30k, MS-COCO to ensure smaller training times(faster feedback) on CPUs & GPUs.

## Performance Measurement Metric: BLEU
BLEU (Bilingual Evaluation Understudy) is a metric for evaluating the quality of generated texts compared to reference texts. It measures n-gram precision and includes a brevity penalty. BLEU scores range from 0 to 1, with 1 indicating a perfect match. It is commonly used in machine translation and image captioning tasks.

## Sample Outputs 
![image](https://github.com/Ash4dev/caption-generator/assets/77205433/49692f14-3826-407e-9c00-158190334a73)
![image](https://github.com/Ash4dev/caption-generator/assets/77205433/117f64dc-6af4-4ed4-9ed7-5fdab04a0dc7)


## Possible Applications 
1. Enhanced accessibility for visually impaired individuals
2. Improved search engine and social media indexing & retreival (greater visibility)
3. Tagline generator for advertisments

## Approach

1. Image Preprocessing:
  - Normalize and resize the images.
  - Prepare them for feature extraction.

2. Feature Extraction:
  - Use a pre-trained CNN architecture (e.g., Xception) to extract meaningful image features.
  - CNN learns visual patterns and encodes them into feature vectors.
  
3. Caption Preprocessing:
  - Clean the captions by removing punctuation, lowercasing, etc.
  - Tokenize the cleaned captions into individual words or tokens.

4. Embedding Generation:
  -  Pass the tokenized & padded captions through a trainable embedding layer.
  -  Map words to dense vector representations (word embeddings).
  - Captures semantic meaning and relationships between words.

5. LSTM Processing:
  - Feed word embeddings into an LSTM layer.
  - LSTM captures sequential information and dependencies in the captions.
  - Learns to predict the likelihood of the next word based on preceding words.

6. Merging and Classification:
  -  Merge image features from the CNN with processed captions from the LSTM.
  - Pass the merged data through a dense layer.
  - Perform multi-label classification to predict the likelihood of each word appearing after the preceding sequence.

## Insights from Hyperparamter tuning
1. Shallow networks perform better.
2. Average pooling produces best results.
3. Smaller batch sizes converge the fastest to suitable results

## Possible Updates:
1. Attention + encoder-decoder architecture: Show, Attend & Tell
