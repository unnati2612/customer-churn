import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from data_preprocessing import load_data, preprocess_data

def train_model():
    
    df = load_data("data/churn.csv")
    df = preprocess_data(df)

    
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    from sklearn.ensemble import RandomForestClassifier

def train_model():
    df = load_data("data/churn.csv")
    df = preprocess_data(df)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # scaling
    
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


# convert back to DataFrame 
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    print("Logistic Regression Accuracy:", lr_acc)
    print("Random Forest Accuracy:", rf_acc)
    

    feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    import shap


    explainer = shap.Explainer(rf_model, X_train)


    shap_values = explainer(X_test, check_additivity=False)


    shap.summary_plot(shap_values, X_test)

    
    
    import joblib

    joblib.dump(rf_model, "models/churn_model.pkl")
    
    return lr_model, rf_model

    


if __name__ == "__main__":
    train_model()
    
    