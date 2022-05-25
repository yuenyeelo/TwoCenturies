# TwoCenturies
## Step 1
### Download sentence embedding at https://tfhub.dev/google/universal-sentence-encoder-multilingual/3
### Save to models
## Step 2
### Update global_options.py for the directory/files 
### Input: base path, glassdoor csv , models , topic 
###
## Step 3
### cd scripts
### python embed_topic_classifier.py 
### Make sure the global_options.py define correcly
### OR you can use the option
####     parser.add_option('-m', '--model', action="store", dest="model", help="Sentence Embedding Model Dir", default=glb.embed_model)
####     parser.add_option('-c', '--csv', action="store", dest="csv", help="GlassDoor Reviews csv", default=glb.gd_csvfile)
####    parser.add_option('-t', '--topic', action="store", dest="topic_file", help="Topic Seedword, excel file", default=glb.topic_file)
####  parser.add_option('-o', '--outputdir', action="store", dest="outputdir", help="Output dir", default=glb.output_dir)
## Step 4
### For each columns (["summary","pros_description", "cons_description","advice_to_management"])
### Output topic probs in output_dir
#### advice_to_management_probs.csv	cons_description_probs.csv	pros_description_probs.csv	summary_probs.csv
