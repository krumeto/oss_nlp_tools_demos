from sentence_transformers import SentenceTransformer
from transformers import pipeline
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

# Define functions for caching
@st.cache_resource
def load_qa_model(model_name = "deepset/roberta-base-squad2"):
    qa_model = pipeline("question-answering", 
                        model = model_name, 
                        tokenizer = model_name)
    return(qa_model)

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

option = st.sidebar.selectbox("Select Search Mode", ("Choose recipe by index", "Enter free text to search", "Try Q&A"))

###########################################
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

###########################################
### Option 2 - Enter a text (recipe or ingredients) and vectorise with SentenceTransformer
elif option == "Enter free text to search":
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
        
###########################################        
### Option 3 - Enter a question and get an answer
elif option == "Try Q&A":
    question = st.sidebar.text_input("Ask a question:")
    # Add an option to use multiple results
    n_hits_to_use = st.sidebar.number_input(label='Number of hits to use', value=1)
    
    if question:
        if len(question) < 100:
            st.subheader(f"Query is **{question}**")
        else:
            st.subheader(f"Query is **{question[:100].strip()}...**")
        
        # vectorise the query and search
        query_embedding = model.encode(question)
        results = semantic_search(query_embedding, embeddings, top_k=n_hits_to_use)

        # loop over results, collect scores and indices
        collect_idx_as_you_loop = []
        scores = []
        # append recipes from top hits
        full_text = ""
        for hit in results[0]:
            idx = hit['corpus_id']
            score = hit['score']
            full_text = full_text + "\n" + recipe_data['full_text'].iloc[idx].strip()
            collect_idx_as_you_loop.append(idx)
            scores.append(score)
        
        # load QA model and get answers
        qa_model = load_qa_model()
        answers = qa_model(
            question = question,
            context = full_text.strip(),         
            top_k = 3, # Get the top 3 answers
            max_answer_len = 50, # up to 50 chars
            max_seq_length = 2000, # answer + query 
            max_question_len = 126, # increase the length of the question from the default 64 
            handle_impossible_answer = True
        )
        
        for idx, answer in enumerate(answers):
            st.write(f"**The {idx+1} answer is {answer['answer']}. Confidence score {answer['score']:.4f}**")
        
        st.write('Based on the following recipes:')
        st.dataframe(recipe_data.iloc[collect_idx_as_you_loop,:].assign(search_score = scores))        
        