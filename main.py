from utils import Utils
from models import Models


if __name__ == '__main__':

    utils = Utils()
    models = Models()

    data = utils.load_from_csv('in/happiness.csv')

    print(data.head())

    X, y = utils.features_target(data, ['country', 'rank'], 'score')

    models.grid_training(X, y)
