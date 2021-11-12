# Tell me a few words and I will tell you who you are

## Abstract

Don't you ever read an article or a book and just wonder what the writer would look like? With the emergence of big data, relationships between the structure of a text and the writer/speaker's attributes can be extracted.

Given the large number of quotations available and their associated speaker, one could explore the correlation between lexical features in speech and the speaker's attributes. Such an analysis would allow us to understand if there is any relationship between the socio-cultural status of an individual and the structure of his speech. The analysis can also be performed temporarily from 2015 to 2020 and reveal information concerning the evolution of particular trends over time. The analysis would allow us to cluster speakers that share similarities in both "lexical features" and "speaker attributes" and extract patterns. This could potentially help people estimate the attributes of a writer/speaker such as his/her ethnicity, educational background, gender, age ...

## Research questions
* How does your socio-cultural background affect the way you speak? 
* Does your nationality, gender, educational background or religion condemn you to talk in a specific way? 
* If so, can an algorithm predict who you are given a set of words you uttered?

## Datasets
* Quotebank data with unique quotes 2015-2020
* Wikidata data with speaker information
* Wikidata QID labels

### Data management
We use Google Collab to load and preprocess the raw Quotebank data because the dataset is very large. 
We access each line of the bz2 and parquet files using the methods provided in the examples.
Large compressed files (such as processed Quotebank data) are stored in the Google Drive. The .parquet files are also stored there for easy access.


## Methods
### Preprocessing
#### Quotebank data 
`preprocessing_notebook.ipynb`

The raw Quotebank data is preprocessed in Google Collab by parsing through each line of the JSON files.
Since we are interested in the way people speak, we do the following:
 * Remove the following columns `['urls','phase','date','numOccurences']`
 * Remove quotes with no speakers
 * Remove quotes with multiple associated QIDs
    * Multiple QIDs imply that there are two speakers with the same name. Since we cannot know which person spoke the quote without further investigation, we remove these rows.
 * Store the probability of the most likely speaker only

The resulting data is stored as a .bz2 file. The JSON file has the following keys:
`['quoteID','quotation','speaker','prob','qid']`

We checked that Quotebank identified a speaker for 65-66% of its quotes. After removing the quotes with non-unique speakers, we retain 46-48% of the original quotes, which totals to approximately 53.7 million quotations. 

#### Speaker data
`exploring_quotes.ipynb`

We also need to extract the information of the speakers. We first read all the parquet files into a dataframe. To reduce the overall of the dataset we need to handle, we remove the speaker information columns that we are not interested in. Thus, we removed the columns `[‘aliases’, ‘lastrevid’, ‘US_congress_bio_ID’, ‘label’, ‘candidacy’, ‘type’]`. We can then merge the preprocessed quotebank data with the speaker data. Since there are around 9 million unique speakers, quotes with speakers that are not in the speakers Wikidata provided are removed in this process. We checked that the quotes removed are a small fraction of all the preprocessed quotes (< 1%).

### Feature extraction
`exploring_quotes.ipynb`

With the cleaned data at hand we are now ready to extract relevant features. We have decided to process the data as follows:

Lexical Features: A processed dataframe called lexical would have the following keys:

`[‘quoteID’,  ‘self_pronouns’, ‘union_pronouns’, ‘other_pronouns’, ‘sentiment_rate’, ‘comparative_rate’, ‘verb_tense’]`

The meaning of each column is described below:
* QuoteID : The identification number of the quotation
* self_pronouns: The number of occurence of pronouns related to the self (See table below)
* Union_pronouns: The number of occurence of pronouns related to the union of the self and the other (See table below)
* Other_pronouns: The number of occurence of pronouns related to the other (See table below)
* Sentiment_rate : The sentiment attached to the quotation. Scalar value in the range [-1,1] where -1 is extremely negative and +1 is extremely positive
* Adjectives: Number of adjectives in the quote
* Comparative_rate: Rate of regular adjectives in comparison to superlatives and comparatives. Scalar value in the range [-1,1] where -1 means only comparatives and superlatives, 1 only regular adjectives.

![Screenshot](images/table.png)

Speaker Features: Speaker features are extracted from Wikidata. The dataframe would include the following columns:

`[‘quoteID’,  ‘gender’, ‘nationality’, ‘religion’, ‘educational_level’, ‘birth_date’, 'occupation']`

Once the data has been processed according to our needs, we are now ready to explore and extract patterns. To do so, a dimensionality reduction framework will be used as a first step. Projecting into a lower dimensional space would allow us to observe and extract patterns more easily. One possible method could be a PCA which takes the aggregated features ‘lexical_feature’ and ‘speaker_feature’ and projects the data points into a 3D or 2D space. However, these features mainly consist of discrete and categorical data, which is not optimal for PCA. We therefore might want to use another method, based on discussion with TA. Patterns can then be extracted using unsupervised clustering techniques to generate groups of data points that share similar features. The analysis of these clusters would allow us to make conclusions concerning the correlation between a speaker and the lexical content used. The analysis would make use of all the different tools learned in the ADA course.

The summary of the whole pipeline is summarized in the schematic below:

![Screenshot](images/Pipeline_ADA.drawio.png)

## Proposed timeline & Project Organization
* Week 6
   * Discussing and selecting topic (all)
* Week 7
   * Raw data cleaning and preprocessing (CJ)
* Week 8
   * Speaker feature generation using Wikidata (RD)
   * Lexical feature generation using NLTK (KS, OH)
   * Exploratory data-analysis (all)
* **Milestone P2, due 23:59 CET, 12 Nov 2021**
* Week 9
   * Feature data manipulation (KS, OH)
   * Dimensionality reduction (all)
* Week 10
   * Clustering data (OH, CJ)
   * Exploring clusters (KS, RD)
* Week 11
   * Begin drafting data story (CJ, RD)
   * Data anlysis (all)
* Week 12
   * Data anlysis (OH, RD)
   * Data visualization (CJ, KS)
* Week 13
   * Final analysis and completion of data story (all)
* **Milestone P3, due 23:59 CET, 17 Dec 2021**
