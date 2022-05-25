
# defualt input directories/files 

path="/Users/yuenyeelo/Documents/Jobs/TwoCenturies/NLP_classifier/"
#gd_csvfile=path+"data/test.csv"
gd_csvfile=path+"data/glassdoor_reviews_selected12.csv"
topic_file=path+"topic/TwoCenturies_Glassdoor_topic_seedwords.xlsx"

# Model downloaded from https://tfhub.dev/google/universal-sentence-encoder-multilingual/3
embed_model=path+"models"
output_dir = path+"output/"

# for classification
topic_file=path+"topic/TwoCenturies_Glassdoor_topic_seedwords.xlsx"

# reviews cols
feat_cols=["summary","pros_description", "cons_description","advice_to_management"]



