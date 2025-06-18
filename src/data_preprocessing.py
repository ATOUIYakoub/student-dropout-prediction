import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, sep=';')

    # Binary target: dropout = 1 if G3 < 10 and absences > 10
    df['dropout'] = ((df['G3'] < 10) & (df['absences'] > 10)).astype(int)

    # Drop columns with IDs/names (not useful for ML)
    df = df.drop(['G1', 'G2', 'G3'], axis=1)

    # Encode categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop('dropout', axis=1)
    y = df['dropout']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
