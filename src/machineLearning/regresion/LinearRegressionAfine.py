from src.machineLearning.IMLBaseModel import IMLBaseModel


class LinearRegressionAffine(IMLBaseModel):

    def fit(self, X, Y):
        # calcular W y guardarlo en el modelo
        # self.model = W
        raise NotImplemented

    def predict(self, X):
        # usar el modelo para predecir Y hat a partir de X y W
        raise NotImplementedError
