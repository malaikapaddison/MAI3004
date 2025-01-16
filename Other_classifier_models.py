import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# models
df = pd.read_csv("D:/Virtual studio/MAI3004 test/all_CT.csv")
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

# Function to evaluate models
def evaluate_models(df, target_column):
    results = []

    # Split data into features (X) and target (y)
    X = df.drop(columns=["Relapse"])
    y = df["Relapse"]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    for model_name, model in models.items():
        # Create a pipeline with scaling (for models like SVM/KNN)
        pipeline = Pipeline([
            ('scaler', StandardScaler()), # Standardize features
            ('model', model)
            ])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Evaluate metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Append results
        results.append({
            "Model": model_name,
            "F1 Score": f1,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall
            })

    # Convert results to a DataFrame and sort by C Score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)

    return results_df

print(evaluate_models(df, "RFS"))