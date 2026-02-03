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

    def test_prepare_data_logic(self):
        """Test if prepare_data drops 'user_id' and creates splits correctly."""
        self.model_instance.load_data()
        self.model_instance.prepare_data(target='premium_subscription')

        # Check if X_train exists
        self.assertIsNotNone(self.model_instance.X_train, "X_train was not created")

        # Check if 'user_id' was dropped (it's in the useless_cols list)
        self.assertNotIn('user_id', self.model_instance.X_train.columns)

        # Check if target is removed from features
        self.assertNotIn('premium_subscription', self.model_instance.X_train.columns)

    def test_model_training_pipeline(self):
        # Test if the training pipeline is constructed and fit.
        self.model_instance.load_data()
        self.model_instance.prepare_data(target='premium_subscription')
        self.model_instance.train_model()

        self.assertIsNotNone(self.model_instance.model, "Model object is None")
        # Check if it has a 'predict' method (meaning it's a valid sklearn estimator)
        self.assertTrue(hasattr(self.model_instance.model, 'predict'))

    def tearDown(self):
        """
        Runs AFTER each test.
        Cleans up the dummy CSV file.
        """
        if os.path.exists(self.test_file):
            os.remove(self.test_file)


if __name__ == '__main__':
    unittest.main()
