import os
import pandas as pd
import zipfile
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
ZIP_NAME = "archive.zip"
UNZIP_DIR = os.path.join(DOWNLOADS_DIR, "ensf444_group7")

def unzip_dataset():
    zip_path = os.path.join(DOWNLOADS_DIR, ZIP_NAME)
    if not os.path.exists(zip_path):
        print("Zip file not found.")
        return False

    if not os.path.exists(UNZIP_DIR):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(UNZIP_DIR)
        print("Extracted to:", UNZIP_DIR)
    return True


def list_csv_files():
    return sorted([f for f in os.listdir(UNZIP_DIR) if f.endswith(".csv")])


def load_csv(file_name):
    path = os.path.join(UNZIP_DIR, file_name)
    df = pd.read_csv(path)

    df.columns = [col.strip() for col in df.columns]
    if "Date" in df.columns:
        df.drop(columns=["Date"], inplace=True)

    return df


def preprocess_dataframe(df):
    df = df.ffill().dropna()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna()


def preprocess_data(df, target):
    df = df.copy()
    df[f"{target}_lag1"] = df[target].shift(1)
    df.dropna(inplace=True)

    X = df[[f"{target}_lag1"]]
    y = df[target]
    return X, y


def evaluate_model(y_test, predictions):
    print("\nModel Performance:")
    print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
    print("MSE:", round(mean_squared_error(y_test, predictions), 2))
    print("R²:", round(r2_score(y_test, predictions), 4))


def plot_results(y_test, predictions, title):
    plt.figure(figsize=(14, 6))
    plt.plot(y_test.values, label="Actual", linewidth=1)
    plt.plot(predictions, label="Predicted", linewidth=1)
    plt.title(title, fontsize=14)
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def run_linear_regression(df, symbol, target):
    X, y = preprocess_data(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    evaluate_model(y_test, predictions)
    plot_results(y_test, predictions, f"{symbol} - Predicting {target} with Linear Regression")

def run_random_forest(df, symbol, target):
    print("Random Forest Regressor not implemented yet.")
    # TODO: Add implementation


def run_knn(df, symbol, target):
    print("K-Nearest Neighbors Regressor not implemented yet.")
    # TODO: Add implementation


def compare_all_models(df, symbol, target):
    print("Model comparison not implemented yet.")
    # TODO: Add implementation


def main():
    print("StockSeer - Predicting Stock Market Trends")

    if not unzip_dataset():
        return

    files = list_csv_files()

    while True:
        print("\nRefer to 'available_stocks.csv' to see valid stock symbols.")
        symbol = input("Enter stock symbol (e.g., AAPL), or type 'exit' to quit: ").strip().upper()

        if symbol.lower() == "exit":
            print("Exiting the program.")
            break

        file_name = f"{symbol}.csv"
        if file_name not in files:
            print("Symbol not found.")
            continue

        df = load_csv(file_name)
        df = preprocess_dataframe(df)

        print("\nColumns available for prediction:")
        print(", ".join(df.columns))

        target = input("Enter the column you want to predict (e.g., Close): ").strip()
        if target not in df.columns:
            print("Invalid column name.")
            continue

        print("\nSelect the model you want to use:")
        print("1. Linear Regression (LR)")
        print("2. Random Forest Regressor (RFR)")
        print("3. K-Nearest Neighbors Regressor (KNN)")
        print("4. Compare all three models")

        choice = input("Enter 1, 2, 3, or 4: ").strip()

        if choice == "1":
            run_linear_regression(df, symbol, target)
        elif choice == "2":
            run_random_forest(df, symbol, target)
        elif choice == "3":
            run_knn(df, symbol, target)
        elif choice == "4":
            compare_all_models(df, symbol, target)
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
