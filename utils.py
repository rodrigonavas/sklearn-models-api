import pandas as pd
import joblib


class Utils:

    def load_from_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def features_target(self, df: pd.DataFrame, drop_cols: list[str], target_name: str) -> pd.DataFrame:
        drop_cols.append(target_name)

        X = df.drop(drop_cols, axis=1)
        y = df[target_name]
        return X, y

    def model_export(self, clf, score):
        print('Score: ', score)
        joblib.dump(clf, 'models/best_model.pkl')
