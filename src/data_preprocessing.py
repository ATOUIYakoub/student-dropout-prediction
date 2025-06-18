import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path, sep=";")


    # Define target: dropout if total grades < 30
    df['dropout'] = ((df['G1'] + df['G2'] + df['G3']) < 30).astype(int)

    selected_features = ['sex', 'age', 'studytime', 'absences', 'G1', 'G2', 'internet']
    X = df[selected_features]
    y = df['dropout']

    X_encoded = pd.get_dummies(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    full_feature_columns = X_encoded.columns

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, full_feature_columns, scaler

def preprocess_single_input(input_data, full_feature_columns, scaler):
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=full_feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    return input_scaled
