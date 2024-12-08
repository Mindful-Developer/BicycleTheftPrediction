import os

from sklearn.model_selection import train_test_split

from data_preprocessing import load_data, preprocess_features, prepare_data_for_training
from model import BicycleTheftModel
import joblib

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'logistic': {'model_type': 'logistic', 'handle_imbalance': True},
        'decision_tree': {'model_type': 'decision_tree', 'handle_imbalance': True},
        'random_forest': {'model_type': 'random_forest', 'handle_imbalance': True},
        'gradient_boosting': {'model_type': 'gradient_boosting', 'handle_imbalance': True},
    }
    
    results = {}
    for name, params in models.items():
        print(f"\nTraining {name}...")
        model = BicycleTheftModel(**params)

        cv_results = model.cross_validate(X_train, y_train)
        print(f"Cross-validation results for {name}:")
        for metric, (mean, std) in cv_results.items():
            print(f"{metric}: {mean:.3f} (+/- {std:.3f})")

        model.fit(X_train, y_train)
        test_results = model.evaluate(X_test, y_test)
        results[name] = {
            'cv_results': cv_results,
            'test_results': test_results,
            'model': model
        }
        
        print(f"Test set results for {name}:")
        for metric, score in test_results.items():
            if isinstance(score, (int, float)):
                print(f"{metric}: {score:.3f}")
            else:
                print(f"{metric}: {score}")
    
    return results

def select_best_model(results):
    best_score = -1
    best_name = None
    best_model = None
    
    for name, result in results.items():
        test_f1 = result['test_results'].get('f1', -1)
        if test_f1 > best_score:
            best_score = test_f1
            best_name = name
            best_model = result['model']
    
    return best_name, best_model, best_score

def main():
    if not os.path.exists('../models'):
        os.makedirs('../models')

    # Load and prepare data
    print("Loading and preparing data...")
    data = load_data('../data/Bicycle_Thefts_Data.csv')
    X, y = prepare_data_for_training(data)  # This will properly convert STATUS to binary

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save feature order for later use
    feature_order = X.columns.tolist()
    joblib.dump(feature_order, '../models/feature_order.pkl')

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    best_name, best_model, best_score = select_best_model(results)
    print(f"\nBest model: {best_name} (F1-score: {best_score:.3f})")
    
    model_path = '../models/best_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved to {model_path}")

if __name__ == '__main__':
    main()
