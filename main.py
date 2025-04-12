# Note: Most of this code was written using material learned in the course and from previous assignments. 
# ChatGPT (OpenAI, 2025) was used to help fix errors,improve some parts of the code, and optimize the machine learning models.

#Please make sure 'archive.zip' is placed in your Downloads folder before running this script.

# Import the needed libraries
import os
import pandas as pd
import zipfile
import matplotlib.pyplot as plt

print("NOTE: Please make sure 'archive.zip' is placed in your Downloads folder before running this script.\n")

# Sklearn stuff for modeling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up folder paths
DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
ZIP_NAME = "archive.zip"
UNZIP_DIR = os.path.join(DOWNLOADS_DIR, "ensf444_group7")

# Unzip the dataset from the Downloads folder
def unzip_dataset():
    zip_path = os.path.join(DOWNLOADS_DIR, ZIP_NAME)
    if not os.path.exists(zip_path):
        print("Zip file not found. Did you download it?")
        return False
    if not os.path.exists(UNZIP_DIR):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(UNZIP_DIR)
        print("Unzipped successfully to:", UNZIP_DIR)
    return True

# Get list of CSV files inside the extracted folder
def list_csv_files():
    return sorted([f for f in os.listdir(UNZIP_DIR) if f.endswith(".csv")])

# Load a specific stock file and drop the Date column
def load_csv(file_name):
    path = os.path.join(UNZIP_DIR, file_name)
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    if "Date" in df.columns:
        df.drop(columns=["Date"], inplace=True)
    return df

# Clean the data — forward fill missing values and convert everything to numbers
def preprocess_dataframe(df):
    df = df.ffill().dropna()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()

# Prepare data for training — create lag feature and target
def preprocess_data(df, target):
    df = df.copy()
    df[f"{target}_lag1"] = df[target].shift(1)
    df.dropna(inplace=True)
    X = df[[f"{target}_lag1"]]
    y = df[target]
    return X, y

# Print and return model evaluation scores
def evaluate_model(y_test, predictions):
    mae = round(mean_absolute_error(y_test, predictions), 2)
    mse = round(mean_squared_error(y_test, predictions), 2)
    r2 = round(r2_score(y_test, predictions), 4)
    print("\nModel Performance:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("R² :", r2)
    return mae, mse, r2

# Show graph with actual vs predicted lines
def plot_results(y_test, predictions_dict, title, metrics_dict=None):
    plt.figure(figsize=(14, 6))
    plt.plot(y_test.values, label="Actual", linewidth=2)
    colors = ['orange', 'green', 'red']
    for (model_name, preds), color in zip(predictions_dict.items(), colors):
        plt.plot(preds, label=f"{model_name} Prediction", linewidth=1.5, color=color)

    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Value ($)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Train and show results for Linear Regression
def run_linear_regression(df, symbol, target):
    X, y = preprocess_data(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    evaluate_model(y_test, predictions)
    plot_results(y_test, {"Linear Regression": predictions}, f"{symbol} - {target} Prediction with Linear Regression")

# Train and show results for Random Forest
def run_random_forest(df, symbol, target):
    X, y = preprocess_data(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])
    param_grid = {
        'rf__n_estimators': [100],
        'rf__max_depth': [None],
        'rf__min_samples_split': [2],
        'rf__min_samples_leaf': [1],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, n_jobs=-1, scoring='neg_mean_squared_error')
    print("\nTraining Random Forest...")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    predictions = best_model.predict(X_test)
    evaluate_model(y_test, predictions)
    plot_results(y_test, {"Random Forest": predictions}, f"{symbol} - {target} Prediction with Random Forest")

# Train and show results for KNN
def run_knn(df, symbol, target):
    X, y = preprocess_data(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor())
    ])
    param_grid = {
        'knn__n_neighbors': [5],
        'knn__weights': ['distance'],
        'knn__metric': ['euclidean']
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, n_jobs=-1, scoring='neg_mean_squared_error')
    print("\nTraining KNN...")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    predictions = best_model.predict(X_test)
    evaluate_model(y_test, predictions)
    plot_results(y_test, {"KNN": predictions}, f"{symbol} - {target} Prediction with KNN")

# Train and compare all 3 models
def compare_all_models(df, symbol, target):
    X, y = preprocess_data(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    tscv = TimeSeriesSplit(n_splits=5)
    models = {
        "Linear Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ]),
        "Random Forest": GridSearchCV(
            Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestRegressor(random_state=42))
            ]),
            param_grid={
                'rf__n_estimators': [100],
                'rf__max_depth': [None],
                'rf__min_samples_split': [2],
                'rf__min_samples_leaf': [1]
            },
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        ),
        "KNN": GridSearchCV(
            Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsRegressor())
            ]),
            param_grid={
                'knn__n_neighbors': [5],
                'knn__weights': ['distance'],
                'knn__metric': ['euclidean']
            },
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
    }
    results = []
    predictions_dict = {}

    # Train each model and collect results
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        best_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        predictions = best_model.predict(X_test)
        predictions_dict[name] = predictions
        mae, mse, r2 = evaluate_model(y_test, predictions)
        results.append({"Model": name, "MAE": mae, "MSE": mse, "R²": r2})

    # Show all metrics in a table and graph
    df_results = pd.DataFrame(results)
    print("\nModel Comparison Table:")
    print(df_results.to_string(index=False))
    metrics_dict = {r['Model']: (r['MAE'], r['MSE'], r['R²']) for r in results}
    plot_results(y_test, predictions_dict, f"{symbol} - {target} Prediction with All Models", metrics_dict)

# Main loop — lets user run everything from the terminal
def main():
    print("StockSeer - Predicting Stock Market Trends")

    if not unzip_dataset():
        return

    files = list_csv_files()

    while True:
        print("\nRefer to 'List.pdf' to see available stock symbols.\n")
        symbol = input("Enter stock symbol (e.g., BTF, SCJ), or type 'exit' to quit: \n").strip().upper()
        if symbol.lower() == "exit":
            print("Exiting the program.")
            break
        file_name = f"{symbol}.csv"
        if file_name not in files:
            print("That stock symbol was not found.")
            continue
        df = load_csv(file_name)
        df = preprocess_dataframe(df)
        print("\nColumns available for prediction:\n")
        print(", ".join(df.columns))
        target = input("Enter column to predict (e.g., Close): \n").strip()
        if target not in df.columns:
            print("Invalid column.")
            continue
        print("\nChoose model:")
        print("1. Linear Regression (LR)")
        print("2. Random Forest Regressor (RFR)")
        print("3. K-Nearest Neighbors Regressor (KNN)")
        print("4. Compare all three models")
        choice = input("Your choice (1-4): ").strip()
        if choice == "1":
            run_linear_regression(df, symbol, target)
        elif choice == "2":
            run_random_forest(df, symbol, target)
        elif choice == "3":
            run_knn(df, symbol, target)
        elif choice == "4":
            compare_all_models(df, symbol, target)
        else:
            print("Invalid option. Please choose from 1 to 4.")

# This starts the program
if __name__ == "__main__":
    main()
