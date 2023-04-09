# oss_nlp_tools_demos

This repo contains demos of Open Source NLP tools

## Dataset

The dataset used is the [Recipe Box](https://eightportions.com/datasets/Recipes/) dataset, collected by Ryan Lee. For the demos, I've kept only recipes with more than 20 words and cleaned the data lightly (see the [preprocess_data.py script](../oss_nlp_tools_demos/data/preprocess_data.py)).

## Setup

### Dev Container (preferred)

The repo could run locally on a virtual environment, but I recommend using the Dev Container setup.

For a dev container setup in VScode, you'd need

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. The Python and Dev Containers VSCode extensions.

    Once installed, check that you see a new icon at the bottom-left of the screen, it should looks like this: `><` with the right bracket a bit higher than the left bracket.

3. Make sure that you have a `.env` file in root that stores #TODO.

4.. Open the repo in the container.

    The next thing to do is to run the Docker container specified in Dockerfile (with Python) and open this repository in that container. To do this, click on the `><` icon bottom-left of the screen and select "Reopen in Container". Once all requirements defined in requirements.txt are installed, the environment is set and you can code forward.

### Virtual Environment

If you prefer to work on a virtual environment, you can do your usual routine, for example. 

```bash
python3 -m venv nlp_tools
source nlp_tools/bin/activate
pip install -r requirements.txt
```

In all cases, run the below once either the Dev Container or the virtual environment is activated, so that the imports work (I hate python imports)

```bash
sudo python setup.py develop
```

## Tools

- [SentenceTransformers](https://www.sbert.net/) - document embeddings creation
- [Brief TfIDF cameo](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) - document embeddings creation
- [Bulk](https://github.com/koaning/bulk) - data viz, data EDA & initial labelling.
- [BERTopic](https://maartengr.github.io/BERTopic/getting_started/quickstart/quickstart.html) - topic modelling & initial labelling
- [Pigeon](https://github.com/agermanidis/pigeon) - simple annotation in Jupyter
- [Langchain & Chroma](https://python.langchain.com/en/latest/index.html) - many NLP goodies, but in this case, indexing, vector storing & search.
- [Simsity](https://github.com/koaning/simsity) - lightweight indexing, storing and search
- [Streamlit](https://streamlit.io/) - simple deployment and data apps