import os
import nltk
from flask import Flask
from collections import namedtuple


app = Flask(__name__)
app.config.from_object(__name__)
Data = namedtuple('Data', 'matrix model dictionary data_frame')
Params = namedtuple('Params', 'user_input score_threshold')
nltk.data.path.append(os.path.join(app.root_path, 'nltk_data/'))
DF_COLUMNS = ['age', 'county', 'date', 'info_link', 'name', 'no', 'race', 'stm', 'stm_link', 'surename', 'county', 'ms',
              'short_stm', 'index', 'date_str', 'nms']

app.config.update(dict(
    SECRET_KEY='\x00\x8ex\x06*BV\xba\x93\x9d-\x9e\xe9\x844\xaa-\xb7\x81&i\xcbJu',
    USERNAME='admin',
    PASSWORD='default',
    MATRIX=os.path.join(app.root_path, 'data/lsi.matrix'),
    MODEL=os.path.join(app.root_path, 'data/lsi.model'),
    DATA_FRAME=os.path.join(app.root_path, 'data/exp2.pkl'),
    DICTIONARY=os.path.join(app.root_path, 'data/dictionary.dict'),
    PHOTOS_PATH=os.path.join(app.root_path, 'static/images')
))
app.config.from_envvar('FLASKR_SETTINGS', silent=True)