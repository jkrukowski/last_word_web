
class Result(object):

    def __init__(self, index, value, data):
        self.index = index
        self.value = value
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
