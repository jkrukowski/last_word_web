import os
from collections import namedtuple
from gensim import corpora, models, similarities
from textblob import TextBlob
import pandas as pd
from flask import Flask, request, g, flash, jsonify, render_template

app = Flask(__name__)
app.config.from_object(__name__)
Data = namedtuple('Data', 'matrix model dictionary data_frame')

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


def load_data():
    """
    Loads dictionary, matrix and model to process data
    :return: namedtuple with dictionary, matrix, model and data frame
    """
    dictionary = corpora.Dictionary.load(app.config['DICTIONARY'])
    matrix = similarities.MatrixSimilarity.load(app.config['MATRIX'])
    model = models.LsiModel.load(app.config['MODEL'])
    df = pd.read_pickle(app.config['DATA_FRAME'])
    return Data(matrix=matrix, model=model, dictionary=dictionary, data_frame=df)


def get_data():
    """
    Attaches dictionary, matrix, model and data frame to global app state
    :return: namedtuple with dictionary, matrix and model
    """
    if not hasattr(g, 'data'):
        g.data = load_data()
    return g.data


def get_record(index, df):
    return df.loc[index].to_dict()


def parse_input(input_data, dictionary, model):
    """
    Parses and transforms user input
    :param input_data: raw text user input
    :param dictionary: gensim dictionary created from corpus
    :param model: gensim lsi model
    :return: user input tranfsormed by gensim model
    """
    vec_text = TextBlob(input_data).words.lower().lemmatize()
    vec_bow = dictionary.doc2bow(vec_text)
    return model[vec_bow]


def get_similar(vec_model, matrix, df):
    """
    Get similar documents
    :param vec_model: user input tranfsormed by gensim model
    :param matrix: gensim similarity matrix
    :return: sorted list of similar documents
    """
    sims = matrix[vec_model]
    result = [{'index': index, 'value': float(item), 'data': get_record(index, df)} for index, item in enumerate(sims)]
    return sorted(result, key=lambda x: -x['value'])


@app.errorhandler(404)
def page_not_found(error):
    return jsonify({'status': 1, 'description': error.description})


@app.route('/query')
def user_query():
    data = get_data()
    user_input = request.args.get('q')
    vec_parsed = parse_input(user_input, data.dictionary, data.model)
    result = get_similar(vec_parsed, data.matrix, data.data_frame)
    flash('successful query')
    return render_template('details.html', result=result)


@app.route('/')
def main_page():
    return render_template('main.html')

if __name__ == '__main__':
    app.run()
