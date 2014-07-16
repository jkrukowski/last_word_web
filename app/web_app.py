from gensim import corpora, models, similarities
from textblob import TextBlob
import pandas as pd
from flask import request, g, escape, redirect, render_template, session, abort
from settings import app, Data, Params, DF_COLUMNS
from os import listdir
from os.path import isfile, join
import random
import plot
import sys


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


def get_photos():
    if not hasattr(g, 'photos'):
        photo_path = app.config['PHOTOS_PATH']
        g.photos = [f for f in listdir(photo_path) if isfile(join(photo_path, f))]
    return g.photos


def choose_photos(n=5):
    return random.sample(get_photos(), n)


def get_data():
    """
    Attaches dictionary, matrix, model and data frame to global app state
    :return: namedtuple with dictionary, matrix and model
    """
    if not hasattr(g, 'data'):
        g.data = load_data()
    return g.data


def get_plot_data():
    """
    Attaches plot data to global app state
    :return: dict with plot data
    """
    if not hasattr(g, 'plot_data'):
        g.plot_data = plot.bar_plot(get_data().data_frame)
    return g.plot_data


def get_record(index, df):
    return df.loc[index, DF_COLUMNS].to_dict()


def get_inmate(inmate_id):
    try:
        data = get_data()
        item = get_record(inmate_id, data.data_frame)
    except KeyError as e:
        item = None
    return item


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


def get_params():
    try:
        user_input = request.args.get('q')
        min_tresh = float(request.args.get('min', 0.001))
        result = Params(user_input=user_input, score_threshold=min_tresh)
    except ValueError as e:
        result = Params(user_input=user_input, score_threshold=0.001)
    return result


def get_similar(vec_model, matrix, df, min_val):
    """
    Get similar documents
    :param vec_model: user input tranfsormed by gensim model
    :param matrix: gensim similarity matrix
    :param df: pandas data frame with data
    :return: sorted DataFrame of similar documents
    """
    sims = matrix[vec_model]
    result_map = {index: float(value) for index, value in enumerate(sims) if value > min_val}
    df['sim'] = pd.Series(result_map)
    df = df.dropna()
    return df.sort('sim', ascending=False)


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html', error_url=request.url)


@app.errorhandler(500)
def page_not_found(e):
    return render_template('500.html', error_url=request.url)


@app.route('/info')
def user_info():
    return render_template('info.html')


@app.route('/query')
def user_query():
    data = get_data()
    params = get_params()
    vec_parsed = parse_input(params.user_input, data.dictionary, data.model)
    result = get_similar(vec_parsed, data.matrix, data.data_frame, params.score_threshold)

    # store session data
    session['user_input'] = params.user_input
    session['score_threshold'] = params.score_threshold

    if not result.empty:
        return render_template('details.html',
                               result=result.to_dict(outtype='records'),
                               user_input=params.user_input,
                               score_threshold=params.score_threshold,
                               all_plot=get_plot_data(),
                               user_plot=plot.bar_plot(result),
                               score_plot=plot.score_plot(result))
    else:
        return render_template('empty.html', user_input=params.user_input)


@app.route('/inmate/<int:inmate_id>')
def inmate_details(inmate_id):
    item = get_inmate(inmate_id)
    data = get_data()
    df = data.data_frame.loc[item['ms']]

    user_input = None
    if 'user_input' in session:
        user_input = session['user_input']
    if item:
        return render_template('single-inmate.html',
                               item=item,
                               user_input=user_input,
                               similar=df.to_dict(outtype='records'))
    else:
        return abort(404)


@app.route('/')
def main_page():
    return render_template('main.html', photos=choose_photos())
