# train

# 默认生成default
python -m train -c sample_configs/config_custom.yml -d data/rasa/movie.json -o models/

python -m train -c sample_configs/config_ltp.yml -d data/rasa/movie.json -o models/ --project ltp


python -m train -c sample_configs/config_ltp.yml -d data/rasa/movie.json -o models/ --project ltp

python -m train -c sample_configs/config_n2g.yml -d data/rasa/movie.json -o models/ --project n2g

python -m train -c sample_configs/config_call_graph.yml -d data/call2graph/rules.md -o models/ --project c2g


# 指定模型的具体位置
python -m train -c sample_configs/config_n2g.yml -d data/rasa/movie.json -o models/ --project n2g --fixed_model_name model

# coref
python -m train -c sample_configs/config_coref.yml -d data/rasa/movie.json -o models/ --project coref


# server
python -m server --path models

# test
python -m unittest tests.call2graph.api_test