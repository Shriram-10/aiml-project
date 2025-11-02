import pickle
import pandas as pd
import os

MODELS_DIR = 'models'

def load_artifacts(models_dir=MODELS_DIR):
    with open(os.path.join(models_dir, 'feature_columns.pkl'), 'rb') as f:
        feature_columns = pickle.load(f)
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(models_dir, 'svm_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    return feature_columns, scaler, model


def prepare_features(df, feature_columns):
    # Keep only expected columns and add missing ones with 0
    X = df.copy()
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]
    return X


def predict(input_csv, output_csv=None):
    feature_columns, scaler, model = load_artifacts()
    df = pd.read_csv(input_csv)
    X = prepare_features(df, feature_columns)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    df['predicted_FTR'] = preds
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Load model artifacts and predict on prepared feature CSV')
    parser.add_argument('input_csv', help='CSV file with features matching feature_columns.pkl')
    parser.add_argument('--output', '-o', help='Optional output CSV to write predictions to')
    args = parser.parse_args()
    result = predict(args.input_csv, args.output)
    print(result[['HomeTeam', 'AwayTeam', 'predicted_FTR']].head())
