#!/bin/bash

# create the data directory if it doesn't exist
mkdir -p data

# download the dataset
wget -P data https://eightportions.com/recipes_raw.zip

# chmod +x dataload.sh to make it executable
# Then ./dataload.sh 