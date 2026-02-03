import unittest
import pandas as pd
import os
from main import ShopperModel  # Importing our class from main.py


class TestShopperModel(unittest.TestCase):

    def setUp(self):
        """
        Runs BEFORE each test.
        Creates a dummy CSV file to avoid testing on the large production dataset.
        """
        self.test_file = 'dummy_test_data.csv'

        # Creating synthetic data with some tricky column names (spaces)
        data = {
            'user_id': list(range(20)),  # Unikalne ID od 0 do 19
            'age': [25, 30, None, 40, 45] * 4,
            'gender': ['Male', 'Female', 'Female', 'Male', 'Female'] * 4,
            'Income Level': [50000, 60000, 70000, 80000, 90000] * 4,
            'premium_subscription': [0, 1, 0, 1, 0] * 4  # Binary Target
        }

        df = pd.DataFrame(data)
        df.to_csv(self.test_file, index=False)

        # Initialize the class
        self.model_instance = ShopperModel(self.test_file)

    def test_load_data_and_normalize_columns(self):
        """Test if data loads and column names are standardized (lowercase + underscores)."""
        self.model_instance.load_data()

        self.assertIsNotNone(self.model_instance.data, "Dataframe is None after loading")
        self.assertEqual(len(self.model_instance.data), 20, "Row count mismatch")

        # Check if 'Income Level' became 'income_level'
        self.assertIn('income_level', self.model_instance.data.columns)
        self.assertNotIn('Income Level', self.model_instance.data.columns)

