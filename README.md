# TwoCenturies NLP project
##### Multi-class topic classifer for GlassDoor reviews. We defined a set of topic seedwords using semi-supervised method. We performanced unsupervised clustering using K-mean clustering and extracted a set of seedwords, we manually defined a set of culture topics based on the clusters and refined the seedwords.

##### The reviews and topic seedwords are embedded using Universal Sentence encoder (see https://tfhub.dev/google/collections/universal-sentence-encoder/1), you can try different sentence-encoder model.  For each review, there are more than one topic. For the initial modeling, we use Support Vector Classification (SVC) and enable probability estimates, the SVC classifier will give the prob of each topic of the review. 

##### Evaluation (performance matric) ... Since there is no ture label of the reviews, 

##### There are still lot of room for improvement...
##

## How To
### Step 1
#### Download sentence embedding at https://tfhub.dev/google/universal-sentence-encoder-multilingual/3
#### Save to models
### Step 2
#### Update global_options.py for the directory/files 
#### Input: base path, glassdoor csv , models , topic 
####
### Step 3
#### cd scripts
#### python embed_topic_classifier.py 
#### Make sure the global_options.py define correcly
#### OR you can use the option
#####     parser.add_option('-m', '--model', action="store", dest="model", help="Sentence Embedding Model Dir", default=glb.embed_model)
#####     parser.add_option('-c', '--csv', action="store", dest="csv", help="GlassDoor Reviews csv", default=glb.gd_csvfile)
#####    parser.add_option('-t', '--topic', action="store", dest="topic_file", help="Topic Seedword, excel file", default=glb.topic_file)
#####  parser.add_option('-o', '--outputdir', action="store", dest="outputdir", help="Output dir", default=glb.output_dir)
### Step 4
#### For each columns (["summary","pros_description", "cons_description","advice_to_management"])
#### Output topic probs in output_dir
##### advice_to_management_probs.csv	cons_description_probs.csv	pros_description_probs.csv	summary_probs.csv


