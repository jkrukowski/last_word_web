from gensim import corpora, models, similarities
from textblob import TextBlob
import pandas as pd
from result import merge_data
from flask import request, g, escape, redirect, render_template, session, abort
from settings import app, Data, Params, DF_COLUMNS
from datetime import datetime
import numpy as np
import sys


def plot_data(df):
    df['year'] = df.date.apply(lambda x: x.year)
    race_series = df.groupby('race').size()
    race_series.order(inplace=True)
    race_series = (race_series / race_series.sum()) * 100.0
    race_series = np.round(race_series, decimals=2)

    lab1 = race_series.index.tolist()
    val1 = race_series.tolist()

    val2, lab2 = np.histogram(df.age, bins=[0, 25, 35, 45, 55, 65, 100])
    val2 = (val2 / float(val2.sum())) * 100.0
    val2 = np.round(val2, decimals=2)

    val3, lab3 = np.histogram(df.year, bins=[0, 1985, 1990, 1995, 2000, 2005, 2010, 2015])
    val3 = (val3 / float(val3.sum())) * 100.0
    val3 = np.round(val3, decimals=2)

    return {
        'race': {'lab': lab1, 'val': val1},
        'age': {'lab': ['..25', '26..35', '36..45', '46..55', '56..65', '66..'], 'val': val2.tolist()},
        'year': {'lab': ['..85', '86..90', '91..95', '96..00', '01..05', '06..10', '11..15'], 'val': val3.tolist()}
    }


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


def get_plot_data():
    """
    Attaches plot data to global app state
    :return: dict with plot data
    """
    if not hasattr(g, 'plot_data'):
        g.plot_data = plot_data(get_data().data_frame)
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
    :return: sorted list of similar documents
    """
    sims = matrix[vec_model]
    result = [merge_data(get_record(index, df),
                         index=index,
                         value=float(val)) for index, val in enumerate(sims) if val > min_val]
    return sorted(result, key=lambda x: -x['value'])


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html', error_url=request.url)


@app.route('/info')
def user_info():
    return render_template('info.html')

@app.route('/query')
def user_query():
    data = get_data()
    params = get_params()
    vec_parsed = parse_input(params.user_input, data.dictionary, data.model)
    result = get_similar(vec_parsed, data.matrix, data.data_frame, params.score_threshold)
    score_plot = {
        'lab': ["" for i in xrange(len(result))],
        'val': [round(i['value'], 3) for i in result]
    }

    # store session data
    session['user_input'] = params.user_input
    session['score_threshold'] = params.score_threshold

    if result:
        # TODO:
        # optimize this, right now is converting data from and to padnas DataFrame
        df = pd.DataFrame(result)
        df.date = df.date.apply(lambda x: datetime.strptime(x, "%m-%d-%Y"))
        return render_template('details.html',
                               result=result,
                               user_input=params.user_input,
                               score_threshold=params.score_threshold,
                               all_plot=get_plot_data(),
                               user_plot=plot_data(df),
                               score_plot=score_plot)
    else:
        return render_template('empty.html', user_input=params.user_input)


@app.route('/inmate/<int:inmate_id>')
def inmate_details(inmate_id):
    item = get_inmate(inmate_id)
    user_input = None
    if 'user_input' in session:
        user_input = session['user_input']
    if item:
        return render_template('single-inmate.html', item=item, user_input=user_input)
    else:
        return abort(404)


@app.route('/')
def main_page():
    return render_template('main.html')
