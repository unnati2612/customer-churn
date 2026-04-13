import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    return df

if __name__ == "__main__":
    df = load_data("data/churn.csv")
    df = preprocess_data(df)
    print(df.head())