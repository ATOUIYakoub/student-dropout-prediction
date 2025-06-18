from src.data_preprocessing import load_and_preprocess
from src.model import train_and_evaluate

if __name__ == "__main__":
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_preprocess("data/student-mat.csv")

    # Train models and evaluate
    train_and_evaluate(X_train, X_test, y_train, y_test)
