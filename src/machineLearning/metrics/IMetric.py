class IMetric(object):
    def __call__(self, target, prediction):
        raise NotImplementedError

    def get_name(self):
        return type(self).__name__
