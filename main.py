import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class ShopperModel:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.encoder = None

    def load_data(self):
        # Load csv and fix column names (replace spaces with underscores)
        self.data = pd.read_csv(self.file_name)
        self.data.columns = self.data.columns.str.replace(' ', '_')
        print(f"Data loaded. Rows count: {self.data.shape[0]}")

    def perform_eda(self):
        # Basic statistical analysis
        if self.data is None:
            return

        print("\n*** Statistics ***")
        print(self.data.describe())

        # Select only numbers for correlation heatmap
        num_data = self.data.select_dtypes(include=[np.number])

        # Drop IDs as they mess up the chart
        cols_to_ignore = ['id', 'user_id', 'Customer_ID', 'year', 'month']
        num_data = num_data.drop(columns=[c for c in cols_to_ignore if c in num_data.columns], errors='ignore')

        if not num_data.empty:
            plt.figure(figsize=(16, 12))
            sns.heatmap(num_data.corr(), annot=True, fmt=".1f", cmap='coolwarm', annot_kws={"size": 8})
            plt.title("Feature Correlation")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            print("No numeric data for plotting.")
            
    def prepare_data(self, target):
        if self.data is None: return

        # Dropping useless columns
        cols_to_drop = ['user_id', 'id', 'customer_id', 'last_purchase_date']
        self.data = self.data.drop(columns=cols_to_drop, errors='ignore')

        # Trimming dataset for faster testing
        if len(self.data) > 50000:
            print("Trimming dataset to 50k rows for testing...")
            self.data = self.data.sample(n=100000, random_state=42)

        if target not in self.data.columns:
            print(f"Error: Column {target} not found")
            return

        print(f"\nTarget variable: {target}")

        # Drop rows with missing target values
        self.data = self.data.dropna(subset=[target])

        # Split features and target
        cols_drop = [target, 'id', 'Customer_ID']
        X = self.data.drop(columns=[c for c in cols_drop if c in self.data.columns], errors='ignore')
        y = self.data[target]

        # Label encoder for target
        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(y)

        # Separate numeric and categorical columns
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = X.select_dtypes(include=['object']).columns

        # Pipeline for numerics (impute missing + scaling)
        num_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Pipeline for text/categorical (impute + one hot encoding)
        cat_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine everything
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipe, num_cols),
                ('cat', cat_pipe, cat_cols)
            ])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Train/test split ready.")
        
    def train_model(self):
        if self.X_train is None: return

        # Default model is Random Forest
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        print("\nStarting training...")
        self.model.fit(self.X_train, self.y_train)
        print("Base model trained.")

    def tune_parameters(self):
        # GridSearch - finding best settings
        if self.model is None: return

        print("\nRunning GridSearch (might take a while)...")

        params = [
            {
                'classifier': [RandomForestClassifier(random_state=42)],
                'classifier__n_estimators': [50, 150],
                'classifier__max_depth': [10, 20]
            },
            {
                # Changed solver to lbfgs to avoid multiclass errors
                'classifier': [LogisticRegression(max_iter=1000, solver='lbfgs')],
                'classifier__C': [0.1, 1.0]
            }
        ]

        gs = GridSearchCV(self.model, params, cv=3, scoring='accuracy', n_jobs=-1)
        gs.fit(self.X_train, self.y_train)

        print(f"Best params: {gs.best_params_}")
        print(f"Best score: {gs.best_score_:.4f}")
        self.model = gs.best_estimator_

