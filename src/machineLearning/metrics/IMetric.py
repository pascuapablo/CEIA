class IMetric(object):
    def __call__(self, target, prediction):
        raise NotImplementedError
