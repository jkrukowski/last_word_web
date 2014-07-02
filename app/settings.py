import os
import nltk
from flask import Flask
from collections import namedtuple


app = Flask(__name__)
app.config.from_object(__name__)
Data = namedtuple('Data', 'matrix model dictionary data_frame')
nltk.data.path.append(os.path.join(app.root_path, 'nltk_data/'))
DF_COLUMNS = ['age', 'county', 'date', 'info_link', 'name', 'no', 'race', 'stm', 'stm_link', 'surename']

app.config.update(dict(
    DEBUG=True,
    SECRET_KEY='\xbe\xf5`\xd7\xb0\xab\x93K;<\xccW\xb7q+\xbd\x1f\xfb\xd7\x95\x8e\xcb\x1e\xb3',
    USERNAME='admin',
    PASSWORD='default',
    MATRIX=os.path.join(app.root_path, 'data/lsi.matrix'),
    MODEL=os.path.join(app.root_path, 'data/lsi.model'),
    DATA_FRAME=os.path.join(app.root_path, 'data/data.pkl'),
    DICTIONARY=os.path.join(app.root_path, 'data/dictionary.dict')
))
app.config.from_envvar('FLASKR_SETTINGS', silent=True)