import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from google.cloud import storage


def model_training():
    data = pd.read_csv('gs://my-data-repo/bank_churn_train.csv')
    data.drop(['id', 'CustomerId', 'Surname'], axis=1, inplace=True)
    data.columns = data.columns.str.lower()
    data.dropna(inplace=True)
    features = ['age', 'numofproducts', 'isactivemember']

    train_x, test_x, train_y, test_y = train_test_split(data[features], data['exited'], test_size=0.2, random_state=101)

    rfc = RandomForestClassifier(n_estimators=10)
    rfc.fit(train_x, train_y)
    predict = rfc.predict(test_x)
    accuracy = accuracy_score(test_y, predict)
    pickle.dump(rfc, open('model.pkl', 'wb'))
    print('Model Saved!')
    storage_client = storage.Client()
    bucket = storage_client.bucket('model-collections-v1')
    blob = bucket.blob('model.pkl')
    blob.upload_from_filename('model.pkl')
    print('Model Pushed!')


if __name__ == "__main__":
    model_training()
