from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from data.preprocess_data import combine_json_to_dataframe

import streamlit as st
from joblib import load

# Define functions for caching
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L12-v2")
    model.max_seq_length = 512
    return(model)

@st.cache_data
def load_data():
    recipe_data = (combine_json_to_dataframe("./data/recipes_raw.zip").
               loc[:, ['title', 'full_text']].
               reset_index(drop=True))
    return(recipe_data)

@st.cache_data
def load_embeddings():
    return load("./embeddings/st_embeddings.joblib")
    
# Set browser title
st.set_page_config(page_title="NLP Tools Demo", page_icon=":robot:")

st.title("The recipe search app :green_salad: :shallow_pan_of_food:")

st.write(":point_left: You can provide a recipe index (first column) and get the top 10 closest recipes")
st.write(":point_left: You can also just provide a description in a text box and get the top hits")

# Pull data
recipe_data = load_data()

# Pull embeddings
embeddings = load_embeddings()

st.dataframe(recipe_data)
model = load_model()

## Provide input options in the side panel

option = st.sidebar.selectbox("Select Search Mode", ("Choose recipe by index", "Enter text to search"))

### Option 1 - choose a recipe by its index
if option == "Choose recipe by index":
    recipe_index = st.sidebar.number_input("Select a recipe index", min_value = 0)
    # Get the respective embedding
    query_embedding = embeddings[recipe_index]
    st.subheader(f"Query recipe is **{recipe_data['title'].iloc[recipe_index]}**")
    
    # Get top results
    results = semantic_search(query_embedding, embeddings, top_k=10)
    
    st.write("**Top 10 similar recipes:**")
    collect_idx_as_you_loop = [] # collect the indices for later dataframe slicing
    
    for hit in results[0]:
        idx = hit['corpus_id']
        # remove the query recipe from the top 10 hits
        if idx == recipe_index:
            pass
        else:
            score = hit['score']
            st.write(f"**{recipe_data['title'].iloc[idx].strip()}** (Score: {score:.4f})")
            collect_idx_as_you_loop.append(idx)
    
    st.write('Here are the recipes:')
    st.dataframe(recipe_data.iloc[collect_idx_as_you_loop,:])

### Option 2 - Enter a text (recipe or ingredients) and vectorise with SentenceTransformer
elif option == "Enter text to search":
    search_text = st.sidebar.text_input("Enter recipe or ingredients")
    # Adding an option for a vector to substract from the main one
    text_to_exclude = st.sidebar.text_input("Ingredients to exclude, if any")
    if search_text and not text_to_exclude:
        if len(search_text) < 100:
            st.subheader(f"Query is **{search_text}**")
        else:
            st.subheader(f"Query is **{search_text[:100].strip()}...**")
        
        # vectorise the query
        query_embedding = model.encode(search_text)
        results = semantic_search(query_embedding, embeddings, top_k=10)

        st.write("**Top 10 similar recipes:**")
        collect_idx_as_you_loop = []
        for hit in results[0]:
            idx = hit['corpus_id']
            score = hit['score']
            st.write(f"**{recipe_data['title'].iloc[idx].strip()}** (Score: {score:.4f})")
            collect_idx_as_you_loop.append(idx)
            
        st.write('Here are the recipes:')
        st.dataframe(recipe_data.iloc[collect_idx_as_you_loop,:])
        
    elif search_text and text_to_exclude:
        if len(search_text) < 100:
            st.subheader(f"Query is **{search_text}**")
        else:
            st.subheader(f"Query is **{search_text[:100].strip()}...**")
        st.subheader(f"Excluding {text_to_exclude}")
        
        #vectorise query and text to exclude
        query_embedding = model.encode(search_text)
        text_to_exclude_embedding = model.encode(text_to_exclude)
        
        # get the results 
        resulting_embedding = query_embedding - text_to_exclude_embedding
        results = semantic_search(resulting_embedding, embeddings, top_k=10)

        st.write("**Top 10 similar recipes:**")
        collect_idx_as_you_loop = []
        for hit in results[0]:
            idx = hit['corpus_id']
            score = hit['score']
            st.write(f"**{recipe_data['title'].iloc[idx].strip()}** (Score: {score:.4f})")
            collect_idx_as_you_loop.append(idx)
            
        st.write('Here are the recipes:')
        st.dataframe(recipe_data.iloc[collect_idx_as_you_loop,:])