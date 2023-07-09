import streamlit as st
import numpy as np

import cv2
import PIL

from models import loadModels, loadTokenizer
from predictions import featExtract, generateSeq

maxLen = 32

def main():
    
    st.title("Image Caption Generator!")
    st.markdown("---")
        
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Go to", ("Home", "Additional Information"))

    if page == "Home":
        st.header("Instructions")
        st.markdown("This Image Caption Generator allows you to upload an image and generate a descriptive caption for it using deep learning techniques.")
        st.markdown("1. Click on the **'Upload an image'** button.")
        st.markdown(
            "2. Select an image file (PNG, JPG, or JPEG) from your local machine.")
        st.markdown("3. Wait for the caption to be generated.")
        st.markdown("4. The generated caption will be displayed below the image.")
        st.markdown("---")
        
        home_page()
    else:
        rest_page()
    
def home_page():
    
    # load models & tokenizers
    ImgModel, FinalModel = loadModels()
    tokenizer = loadTokenizer()
    
    # upload image
    uploaded_file = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        
        img = PIL.Image.open(uploaded_file)
        st.image(img)
        st.markdown("---")
        
        img = PIL.ImageOps.fit(img, (299, 299), PIL.Image.ANTIALIAS)
        
        img  = np.array(img)
                
        # extract features in the image
        extractedFeats = featExtract(img, ImgModel)

        # generating caption
        pred = generateSeq(tokenizer, maxLen, extractedFeats, FinalModel)
        
        st.subheader("Generated Caption:")
        st.write(pred)
        
        st.markdown("---")
        st.markdown("Made by Ashutosh Panigrahy")

def rest_page():

     # Add description and explanation
    st.header("About")
    st.markdown("This Image Caption Generator is made by Ashutosh Panigrahy.")
    st.markdown(
        "The project is based on caption generation using the Flicker8k dataset from Kaggle.")
    st.markdown("The Flicker8k dataset consists of 8,100 images, each accompanied by 5 captions. In this project, approximately 80% (6,400 images) of the dataset was used for training, while the remaining 20% was used for testing.")
    st.markdown("The model generates captions relevant to the visual content of the images by combining image features and linguistic context. This approach utilizes encoder-decoder deep learning architectures (CNNs and LSTMs) to generate meaningful captions for images.")
    st.markdown("The performance of the generated captions can be evaluated using the BLEU (Bilingual Evaluation Understudy) metric, which measures n-gram precision and includes a brevity penalty. Higher BLEU scores indicate better match with reference texts.")

    st.markdown("---")
    st.header("Possible Applications")
    st.markdown("- Enhanced accessibility for visually impaired individuals")
    st.markdown(
        "- Improved search engine and social media indexing & retrieval (greater visibility)")
    st.markdown("- Tagline generator for advertisements")

    st.markdown("---")
    st.header("Approach")
    st.markdown(
        "1. **Image Preprocessing:** Normalize and resize the images. Prepare them for feature extraction.")
    st.markdown("2. **Feature Extraction:** Use a pre-trained CNN architecture (e.g., Xception) to extract meaningful image features. CNN learns visual patterns and encodes them into feature vectors.")
    st.markdown("3. **Caption Preprocessing:** Clean the captions by removing punctuation, lowercasing, etc. Tokenize the cleaned captions into individual words or tokens.")
    st.markdown("4. **Embedding Generation:** Pass the tokenized & padded captions through a trainable embedding layer. Map words to dense vector representations (word embeddings). Captures semantic meaning and relationships between words.")
    st.markdown("5. **LSTM Processing:** Feed word embeddings into an LSTM layer. LSTM captures sequential information and dependencies in the captions. Learns to predict the likelihood of the next word based on preceding words.")
    st.markdown("6. **Merging and Classification:** Merge image features from the CNN with processed captions from the LSTM. Pass the merged data through a dense layer. Perform multi-label classification to predict the likelihood of each word appearing after the preceding sequence.")

    st.markdown("---")
    st.header("Insights from Hyperparameter Tuning")
    st.markdown("- Shallow networks perform better.")
    st.markdown("- Average pooling produces the best results.")
    st.markdown(
        "- Smaller batch sizes converge the fastest to suitable results.")

    st.markdown("---")
    st.header("Possible Updates")
    st.markdown("Consider incorporating attention mechanism and an encoder-decoder architecture using techniques like 'Show, Attend & Tell' for further improving caption generation.")

    st.markdown("---")
    st.markdown("Made by Ashutosh Panigrahy")
    st.markdown("Notebook @ : [kaggle](https: // www.kaggle.com/code/ashutoshpanigrahy/notebook0ecb3cce5b)")
    
if __name__ == "__main__":
    main()