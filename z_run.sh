# train
python -m rasa.nlu.train -c sample_configs/config_custom.yml -d data/movie/movie.json -o models/



# server
python -m rasa.nlu.server --path models