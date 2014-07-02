
format_map = {
    'date': lambda x: x.date(),
    'score': lambda x: "{:.3f}".format(x)
}


def format_value(key, value):
    fun = format_map.get(key, lambda x: x)
    return fun(value)


class Result(object):

    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, format_value(key, dictionary[key]))
        for key in kwargs:
            setattr(self, key, format_value(key, kwargs[key]))
