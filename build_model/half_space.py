import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

columns = [
    'Mean_IP',          'SD_IP',            'Kurtosis_IP',
    'Skewness_IP',      'Mean_DMSNR',       'SD_DMSNR',
    'Kurtosis_DMSNR',   'Skewness_DMSNR',   'Class'
]

dataset = pd.read_csv('../archive/HTRU_2.csv', header=None, names=columns)

y = dataset['Class']
x = dataset.drop(columns=['Class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

param_list = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

def train(feats, target):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_list,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(feats, target)
    return grid.best_estimator_

def evaluate(model, feats, target):
    pred = model.predict(feats)
    print(f"Confusion matrix:\n{confusion_matrix(target, pred)}")
    print(f"Classification report:\n{classification_report(target, pred)}")

def save(model):
    joblib.dump(model, '../models/half-space-mind.pkl')

if __name__ == "__main__":
    half_space = train(x_train, y_train)
    evaluate(half_space, x_test, y_test)
    save(half_space)