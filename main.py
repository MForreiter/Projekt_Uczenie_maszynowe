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

    