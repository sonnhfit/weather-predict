import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#đọc dữ liệu
data = pd.read_csv('./dataset/daily_weather.csv')

data[data.isnull().any(axis=1)]

del data['number']

before_rows = data.shape[0]
#print(before_rows)
data = data.dropna()
after_rows = data.shape[0]
#print(after_rows)
clean_data = data.copy()
clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm'] > 24.99)*1
#print(clean_data['high_humidity_label'])

y=clean_data[['high_humidity_label']].copy()

clean_data['relative_humidity_3pm'].head()

y.head()

morning_features = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',
        'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',
        'rain_duration_9am']

X = clean_data[morning_features].copy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)

humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
humidity_classifier.fit(X_train, y_train)

predictions = humidity_classifier.predict(X_test)
#
# import pickle
# with open('mod.plk', 'wb') as f:
#     pickle.dump(humidity_classifier, f)

a = accuracy_score(y_true = y_test, y_pred = predictions)
print(a)