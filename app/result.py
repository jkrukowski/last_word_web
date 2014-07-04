
format_map = {
    'date': lambda x: x.strftime("%m-%d-%Y"),
    'stm': lambda x: x[:50] + '...' if len(x) > 50 else x
}


def format_value(key, value):
    fun = format_map.get(key, lambda x: x)
    return fun(value)


def merge_data(*initial_data, **kwargs):
    result = {}
    for dictionary in initial_data:
            for key in dictionary:
                result[key] = format_value(key, dictionary[key])
    for key in kwargs:
        result[key] = format_value(key, kwargs[key])
    return result


class Result(object):

    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, format_value(key, dictionary[key]))
        for key in kwargs:
            setattr(self, key, format_value(key, kwargs[key]))
