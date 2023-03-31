import pandas as pd
import json
import zipfile

def combine_json_to_dataframe(zip_file_path, num_words_cutoff = 20) -> pd.DataFrame:
    """
    Combines data from three JSON files containing recipes, loads them into a Pandas DataFrame, and 
    pre-processes the data for further analysis. Returns the resulting DataFrame.
    
    Args:
        zip_file_path (str): The file path of the zip file containing the three JSON files with recipe data.
        num_words_cutoff (int): The minimum number of words in a recipe to be included in the resulting DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing the combined data from the three JSON files with the 
        following columns: 'title', 'ingredients', 'instructions', 'full_text', and 'num_words'. All 
        recipes with fewer than 'num_words_cutoff' words are excluded.
    """
    # Open the zip file and load the JSON files
    with zipfile.ZipFile(zip_file_path) as z:
        with z.open('recipes_raw_nosource_fn.json') as f:
            fn_data = json.load(f)
        with z.open('recipes_raw_nosource_epi.json') as f:
            epi_data = json.load(f)
        with z.open('recipes_raw_nosource_ar.json') as f:
            ar_data = json.load(f)

    # Combine the data from the three JSON files
    data = {**fn_data, **epi_data, **ar_data}

    # Convert the data to a dataframe
    df = pd.DataFrame.from_dict(data, orient='index')

    # Add a new column with the concatenated text
    df['full_text'] = 'Recipe title: ' + df['title'] + '. Ingredients: ' + df['ingredients'].apply(lambda x: '; '.join(x)) + '. Instructions: ' + df['instructions']
    
    # Drop picture link
    df = df.drop(['picture_link'], axis = 1)
    
    # Add a column with the number of words in the full_text column
    df['num_words'] = df['full_text'].str.split().str.len()
    
    # Remove short recipes - most are just tests
    df = df.loc[lambda d: d['num_words'] > num_words_cutoff]

    return df

if __name__ == "__main__":
    full_data = combine_json_to_dataframe("data/recipes_raw.zip")
    print(full_data)

