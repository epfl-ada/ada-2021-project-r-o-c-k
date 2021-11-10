# Tell me a few words I will tell you who you are
# Tell me who you are and I will tell you how you speak

## Abstract

## Research questions

## Datasets
* Quotebank data with unique quotes 2015-2020
* Provided Wikidata data with speaker information

### Data management
We use Google Collab to load and preprocess the raw Quotebank data because the dataset is very large. 
We access each line of the .bz2 and .parquet files using the methods provided in the examples.
Large compressed files (such as processed Quotebank data) are stored in the Google Drive. The .parquet files are also stored there for easy access.


## Methods
### Preprocessing
The raw Quotebank data is preprocessed in Google Collab by parsing through each line of the JSON files.
Since we are interested in the way people speak, we do the following:
 * Remove the following columns ['urls','phase','date','numOccurences']
 * Remove quotes with no speakers
 * Remove quotes with multiple associated QIDs
  * Multiple QIDs implies that there are two speakers with the same name. Since we cannot know which person spoke the quote without further investigation, we remove these rows.
 * Store the probability of the most likely speaker only

The resulting data is stored as a .bz2 file. The JSON file has the following keys:
['quoteID','quotation','speaker','prob','qid']

## Proposed timeline

## Project organization

## Questions for TA (optional)
