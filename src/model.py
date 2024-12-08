import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, make_scorer
from sklearn.model_selection import cross_val_score, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize, StandardScaler
from xgboost import XGBClassifier

class BicycleTheftModel:
    def __init__(self, model_type='random_forest', class_weight='balanced', handle_imbalance=True):
        self.model_type = model_type
        self.handle_imbalance = handle_imbalance
        self.scaler = StandardScaler()
        
        if model_type == 'logistic':
            self.base_model = LogisticRegression(
                class_weight=class_weight,
                max_iter=5000,  
                solver='saga',
                random_state=42,
                n_jobs=-1  
            )
        elif model_type == 'decision_tree':
            self.base_model = DecisionTreeClassifier(class_weight=class_weight, random_state=42)
        elif model_type == 'random_forest':
            self.base_model = RandomForestClassifier(
                class_weight=class_weight,
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1  
            )
        elif model_type == 'gradient_boosting':
            self.base_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if handle_imbalance:
            self.model = ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('classifier', self.base_model)
            ])
        else:
            self.model = self.base_model
    
    def fit(self, X, y):
        # Scale features for logistic regression
        if self.model_type == 'logistic':
            X = self.scaler.fit_transform(X)
            
        if self.handle_imbalance:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            self.model.fit(X_resampled, y_resampled)
        else:
            self.model.fit(X, y)
        return self
    
    def train(self, X_train, y_train):
        """Train the model."""
        self.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        if self.model_type == 'logistic':
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.model_type == 'logistic':
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation and return scores."""
        if self.handle_imbalance:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            X, y = X_resampled, y_resampled
            
        scoring = {
            'accuracy': 'accuracy',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'f1_weighted': 'f1_weighted'
        }
        
        cv_results = cross_validate(self.model, X, y, cv=cv, scoring=scoring)
        
        return {
            'accuracy': (float(np.mean(cv_results['test_accuracy'])), float(np.std(cv_results['test_accuracy']))),
            'precision': (float(np.mean(cv_results['test_precision_weighted'])), float(np.std(cv_results['test_precision_weighted']))),
            'recall': (float(np.mean(cv_results['test_recall_weighted'])), float(np.std(cv_results['test_recall_weighted']))),
            'f1': (float(np.mean(cv_results['test_f1_weighted'])), float(np.std(cv_results['test_f1_weighted'])))
        }
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        results = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted')),
            'f1': float(f1_score(y_test, y_pred, average='weighted')),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }

        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            results['roc_auc'] = auc(fpr, tpr)
            results['fpr'] = fpr
            results['tpr'] = tpr
        else:
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            classes = np.unique(y_test)
            y_test_bin = label_binarize(y_test, classes=classes)

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fpr["micro"], tpr["micro"], _ = roc_curve(
                y_test_bin.ravel(), 
                y_pred_proba.ravel()
            )
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            results['roc_auc'] = roc_auc
            results['fpr'] = fpr
            results['tpr'] = tpr

        return results

    def plot_learning_curves(self, X_test, y_test):
        """Plot ROC and Precision-Recall curves."""
        metrics = self.evaluate(X_test, y_test)
        y_pred_proba = self.predict_proba(X_test)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # ROC curve
        ax1.plot(metrics['fpr'], metrics['tpr'],
                 label=f'ROC curve (AUC = {metrics["roc_auc"]:.2f})')
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()

        # Precision-Recall curve
        # For binary classification, we want probabilities of the positive class (RECOVERED)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        pr_auc = auc(recall, precision)

        ax2.plot(recall, precision,
                 label=f'PR curve (AUC = {pr_auc:.2f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()

        plt.tight_layout()
        return fig
    
    def save_model(self, filepath):
        """Save the trained model."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    @staticmethod
    def load_model(filepath):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance if the model supports it."""
        if self.model_type in ['decision_tree', 'random_forest']:
            if self.handle_imbalance:
                importance = self.model.named_steps['classifier'].feature_importances_
            else:
                importance = self.model.feature_importances_
            
            return dict(zip(feature_names, importance))
        elif self.model_type == 'logistic':
            if self.handle_imbalance:
                coef = self.model.named_steps['classifier'].coef_[0]
            else:
                coef = self.model.coef_[0]
            
            return dict(zip(feature_names, np.abs(coef)))
        else:
            return None
