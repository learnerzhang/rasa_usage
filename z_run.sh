# train

# 默认生成default
python -m rasa.nlu.train -c sample_configs/config_custom.yml -d data/rasa/movie.json -o models/

python -m rasa.nlu.train -c sample_configs/config_ltp.yml -d data/rasa/movie.json -o models/ --project ltp


python -m rasa.nlu.train -c sample_configs/config_ltp.yml -d data/rasa/movie.json -o models/ --project ltp

python -m rasa.nlu.train -c sample_configs/config_n2g.yml -d data/rasa/movie.json -o models/ --project n2g

# 指定模型的具体位置
python -m rasa.nlu.train -c sample_configs/config_n2g.yml -d data/rasa/movie.json -o models/ --project n2g --fixed_model_name model

# coref
python -m rasa.nlu.train -c sample_configs/config_coref.yml -d data/rasa/movie.json -o models/ --project coref


# server
python -m rasa.nlu.server --path models

python -m rasa.nlu.server --path models