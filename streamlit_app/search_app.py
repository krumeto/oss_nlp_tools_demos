Complete the code for the below streamlit app


from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from data.preprocess_data import combine_json_to_dataframe

import streamlit as st
from joblib import load

## Read and display the recipes

# Pull data
recipe_data = combine_json_to_dataframe("../data/recipes_raw.zip")

#  #   Column        Non-Null Count   Dtype  
# ---  ------        --------------   -----  
#  0   instructions  124434 non-null  object 
#  1   ingredients   124434 non-null  object 
#  2   title         124434 non-null  object 
#  3   full_text     124434 non-null  object 
#  4   num_words     124434 non-null  float64

# Pull embeddings
embeddings = load("../embeddings/st_embeddings.joblib")

st.dataframe(recipe_data)

## Provide input options in the side panel

### Option 1 - choose a recipe by its index

### Option 2 - Enter a text (recipe or ingredients) and vectorise with SentenceTransformer
model = SentenceTransformer(""all-MiniLM-L12-v2"")
model.max_seq_length = 512


## Search with semantic_search against the entry of either option 1 or option 2


## Display Top 10 results

## Provide input options in the side panel

option = st.sidebar.selectbox("Select Option", ("Choose recipe by index", "Enter text to search"))

if option == "Choose recipe by index":
    recipe_index = st.sidebar.slider("Select a recipe index", 0, len(recipe_data)-1, 0)
    query_embedding = embeddings[recipe_index]
    results = semantic_search(query_embedding, embeddings, top_k=10)
    st.write("Top 10 similar recipes:")
    for idx, score in results[0]:
        st.write(f"{recipe_data['title'].iloc[idx]} (Score: {score:.4f})")

elif option == "Enter text to search":
    search_text = st.sidebar.text_input("Enter recipe or ingredients")
    if search_text:
        query_embedding = model.encode(search_text)
        results = semantic_search(query_embedding, embeddings, top_k=10)
        st.write("Top 10 similar recipes:")
        for idx, score in results[0]:
            st.write(f"{recipe_data['title'].iloc[idx]} (Score: {score:.4f})")