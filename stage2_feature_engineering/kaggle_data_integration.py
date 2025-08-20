"""
Kaggle Data Integration Module
Fetches and processes real credit scoring datasets from Kaggle for realistic model training.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os
import zipfile
from pathlib import Path
import requests
import json

logger = logging.getLogger(__name__)

class KaggleDataIntegration:
    """Integrates Kaggle datasets for credit intelligence training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kaggle_username = config.get('kaggle_username')
        self.kaggle_key = config.get('kaggle_key')
        self.data_cache_path = Path(config.get('data_cache_path', './kaggle_data/'))
        self.data_cache_path.mkdir(exist_ok=True)
        
        # Available credit datasets
        self.credit_datasets = {
            'credit_score_classification': {
                'id': 'parisrohan/credit-score-classification',
                'description': 'Credit score classification dataset with financial features',
                'features': ['annual_income', 'monthly_inhand_salary', 'num_bank_accounts', 'num_credit_card', 
                           'interest_rate', 'num_of_loan', 'delay_from_due_date', 'num_of_delayed_payment',
                           'changed_credit_limit', 'num_credit_inquiries', 'credit_mix', 'outstanding_debt',
                           'credit_utilization_ratio', 'credit_history_age', 'payment_of_min_amount',
                           'total_emi_per_month', 'amount_invested_monthly', 'payment_behaviour',
                           'monthly_balance'],
                'target': 'credit_score',
                'target_mapping': {'Poor': 0, 'Standard': 1, 'Good': 2}
            },
            'credit_card_fraud': {
                'id': 'mlg-ulb/creditcardfraud',
                'description': 'Credit card fraud detection with anonymized features',
                'features': [f'V{i}' for i in range(1, 29)] + ['Amount'],
                'target': 'Class',
                'target_mapping': {0: 0, 1: 1}  # Binary: 0=Normal, 1=Fraud
            },
            'loan_prediction': {
                'id': 'shivamb/real-or-fake-fake-jobposting-prediction',
                'description': 'Loan prediction dataset with borrower characteristics',
                'features': ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                           'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                           'Credit_History', 'Property_Area'],
                'target': 'Loan_Status',
                'target_mapping': {'Y': 1, 'N': 0}  # Binary: 1=Approved, 0=Rejected
            },
            'german_credit': {
                'id': 'uciml/german-credit-data',
                'description': 'German credit dataset with banking features',
                'features': ['checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
                           'savings_account', 'employment', 'installment_rate', 'personal_status',
                           'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
                           'housing', 'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker'],
                'target': 'credit_risk',
                'target_mapping': {'good': 0, 'bad': 1}  # Binary: 0=Good, 1=Bad
            }
        }
        
    async def initialize(self):
        """Initialize Kaggle data integration"""
        logger.info("Initializing Kaggle data integration...")
        
        if not self.kaggle_username or not self.kaggle_key:
            logger.warning("Kaggle credentials not provided - using cached datasets if available")
            logger.info("To download fresh data, set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        
        # Check if we have cached datasets
        cached_datasets = self._get_cached_datasets()
        if cached_datasets:
            logger.info(f"Found {len(cached_datasets)} cached datasets: {list(cached_datasets.keys())}")
        
        logger.info("Kaggle data integration initialized successfully")
    
    def _get_cached_datasets(self) -> Dict[str, Path]:
        """Get list of cached datasets"""
        cached = {}
        for dataset_name in self.credit_datasets.keys():
            dataset_path = self.data_cache_path / dataset_name
            if dataset_path.exists() and (dataset_path / 'data.csv').exists():
                cached[dataset_name] = dataset_path
        return cached
    
    async def download_dataset(self, dataset_name: str, force_download: bool = False) -> Optional[Path]:
        """Download a specific dataset from Kaggle"""
        if dataset_name not in self.credit_datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            return None
        
        dataset_info = self.credit_datasets[dataset_name]
        dataset_path = self.data_cache_path / dataset_name
        
        # Check if already downloaded
        if not force_download and dataset_path.exists() and (dataset_path / 'data.csv').exists():
            logger.info(f"Dataset {dataset_name} already cached")
            return dataset_path
        
        if not self.kaggle_username or not self.kaggle_key:
            logger.error("Kaggle credentials required for downloading")
            return None
        
        try:
            logger.info(f"Downloading dataset: {dataset_name}")
            
            # Create dataset directory
            dataset_path.mkdir(exist_ok=True)
            
            # Download using Kaggle API (if available) or direct download
            success = await self._download_via_kaggle_api(dataset_name, dataset_info, dataset_path)
            
            if success:
                logger.info(f"Successfully downloaded {dataset_name}")
                return dataset_path
            else:
                logger.error(f"Failed to download {dataset_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {e}")
            return None
    
    async def _download_via_kaggle_api(self, dataset_name: str, dataset_info: Dict, dataset_path: Path) -> bool:
        """Download dataset using Kaggle API"""
        try:
            # Set Kaggle credentials
            os.environ['KAGGLE_USERNAME'] = self.kaggle_username
            os.environ['KAGGLE_KEY'] = self.kaggle_key
            
            # Try to use kaggle CLI if available
            import subprocess
            
            # Download dataset
            cmd = f"kaggle datasets download -d {dataset_info['id']} -p {dataset_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Extract zip file
                zip_files = list(dataset_path.glob("*.zip"))
                if zip_files:
                    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                        zip_ref.extractall(dataset_path)
                    
                    # Remove zip file
                    zip_files[0].unlink()
                    
                    # Find the main data file
                    csv_files = list(dataset_path.glob("*.csv"))
                    if csv_files:
                        # Rename to standard name
                        csv_files[0].rename(dataset_path / 'data.csv')
                        return True
                
                logger.warning(f"No CSV files found in {dataset_name}")
                return False
            else:
                logger.error(f"Kaggle CLI error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error using Kaggle API: {e}")
            return False
    
    async def get_training_dataset(self, dataset_name: str = 'credit_score_classification', 
                                 sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Get training dataset from specified Kaggle dataset"""
        logger.info(f"Loading training dataset: {dataset_name}")
        
        # Download dataset if not cached
        dataset_path = await self.download_dataset(dataset_name)
        if not dataset_path:
            logger.error(f"Could not load dataset: {dataset_name}")
            return pd.DataFrame(), pd.Series()
        
        # Load data
        data_file = dataset_path / 'data.csv'
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return None, None
        
        try:
            # Load data with appropriate encoding
            df = pd.read_csv(data_file, encoding='utf-8')
            
            # Handle different encodings if UTF-8 fails
            if df.empty:
                df = pd.read_csv(data_file, encoding='latin-1')
            
            logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
            
            # Process the dataset
            X, y = self._process_dataset(df, dataset_name, sample_size)
            
            if X is not None and len(X) > 0:
                logger.info(f"Processed dataset: {len(X)} samples, {len(X.columns)} features")
                return X, y
            else:
                logger.error("Failed to process dataset")
                return pd.DataFrame(), pd.Series()
                
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return pd.DataFrame(), pd.Series()
    
    def _process_dataset(self, df: pd.DataFrame, dataset_name: str, 
                        sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Process raw dataset into training format"""
        dataset_info = self.credit_datasets[dataset_name]
        
        try:
            # Sample data if requested
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                logger.info(f"Sampled {sample_size} rows from dataset")
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Encode categorical variables
            df = self._encode_categorical_variables(df, dataset_name)
            
            # Prepare features and target
            X, y = self._prepare_features_and_target(df, dataset_info)
            
            # Feature engineering
            X = self._engineer_features(X, dataset_name)
            
            # Final cleaning
            X = self._final_cleaning(X)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            return None, None
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        # Check missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found: {missing_counts.sum()}")
            
            # For numerical columns, fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            # For categorical columns, fill with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Encode categorical variables"""
        logger.info("Encoding categorical variables...")
        
        # Get dataset info
        dataset_info = self.credit_datasets[dataset_name]
        
        # Handle target variable encoding
        if dataset_info['target'] in df.columns:
            target_col = dataset_info['target']
            target_mapping = dataset_info['target_mapping']
            
            # Apply target mapping
            if target_col in df.columns:
                df[target_col] = df[target_col].map(target_mapping)
                logger.info(f"Encoded target variable: {target_col}")
        
        # Handle other categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != dataset_info['target']:  # Don't encode target
                # Use label encoding for simplicity
                df[col] = df[col].astype('category').cat.codes
        
        return df
    
    def _prepare_features_and_target(self, df: pd.DataFrame, 
                                   dataset_info: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variables"""
        target_col = dataset_info['target']
        
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found in dataset")
            return None, None
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        logger.info(f"Prepared {len(X.columns)} features and target variable")
        return X, y
    
    def _engineer_features(self, X: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Engineer additional features based on dataset type"""
        logger.info("Engineering additional features...")
        
        if dataset_name == 'credit_score_classification':
            X = self._engineer_credit_score_features(X)
        elif dataset_name == 'credit_card_fraud':
            X = self._engineer_fraud_detection_features(X)
        elif dataset_name == 'loan_prediction':
            X = self._engineer_loan_features(X)
        elif dataset_name == 'german_credit':
            X = self._engineer_german_credit_features(X)
        
        return X
    
    def _engineer_credit_score_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for credit score classification"""
        # Add ratio features
        if 'annual_income' in X.columns and 'monthly_inhand_salary' in X.columns:
            X['income_to_salary_ratio'] = X['annual_income'] / (X['monthly_inhand_salary'] * 12)
        
        if 'outstanding_debt' in X.columns and 'annual_income' in X.columns:
            X['debt_to_income_ratio'] = X['outstanding_debt'] / X['annual_income']
        
        if 'total_emi_per_month' in X.columns and 'monthly_inhand_salary' in X.columns:
            X['emi_to_salary_ratio'] = X['total_emi_per_month'] / X['monthly_inhand_salary']
        
        # Add interaction features
        if 'num_credit_card' in X.columns and 'num_of_loan' in X.columns:
            X['total_credit_products'] = X['num_credit_card'] + X['num_of_loan']
        
        # Add risk indicators
        if 'delay_from_due_date' in X.columns:
            X['payment_delay_risk'] = (X['delay_from_due_date'] > 30).astype(int)
        
        if 'num_of_delayed_payment' in X.columns:
            X['delayed_payment_risk'] = (X['num_of_delayed_payment'] > 5).astype(int)
        
        return X
    
    def _engineer_fraud_detection_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for fraud detection"""
        # Add statistical features for V columns
        v_columns = [col for col in X.columns if col.startswith('V')]
        
        if v_columns:
            # Add mean and std of V features
            X['v_features_mean'] = X[v_columns].mean(axis=1)
            X['v_features_std'] = X[v_columns].std(axis=1)
            X['v_features_max'] = X[v_columns].max(axis=1)
            X['v_features_min'] = X[v_columns].min(axis=1)
        
        # Add amount-based features
        if 'Amount' in X.columns:
            X['amount_log'] = np.log1p(X['Amount'])
            X['amount_risk'] = (X['Amount'] > X['Amount'].quantile(0.95)).astype(int)
        
        return X
    
    def _engineer_loan_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for loan prediction"""
        # Add income-based features
        if 'ApplicantIncome' in X.columns:
            X['income_log'] = np.log1p(X['ApplicantIncome'])
            X['high_income'] = (X['ApplicantIncome'] > X['ApplicantIncome'].quantile(0.8)).astype(int)
        
        if 'ApplicantIncome' in X.columns and 'CoapplicantIncome' in X.columns:
            X['total_income'] = X['ApplicantIncome'] + X['CoapplicantIncome']
            X['total_income_log'] = np.log1p(X['total_income'])
        
        # Add loan-based features
        if 'LoanAmount' in X.columns and 'Loan_Amount_Term' in X.columns:
            X['monthly_payment'] = X['LoanAmount'] / X['Loan_Amount_Term']
        
        if 'LoanAmount' in X.columns and 'ApplicantIncome' in X.columns:
            X['loan_to_income_ratio'] = X['LoanAmount'] / X['ApplicantIncome']
        
        return X
    
    def _engineer_german_credit_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for German credit dataset"""
        # Add age-based features
        if 'age' in X.columns:
            X['age_group'] = pd.cut(X['age'], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3])
            X['young_borrower'] = (X['age'] < 30).astype(int)
            X['senior_borrower'] = (X['age'] > 60).astype(int)
        
        # Add duration-based features
        if 'duration' in X.columns:
            X['long_term_credit'] = (X['duration'] > 24).astype(int)
            X['short_term_credit'] = (X['duration'] < 12).astype(int)
        
        # Add amount-based features
        if 'credit_amount' in X.columns:
            X['high_credit_amount'] = (X['credit_amount'] > X['credit_amount'].quantile(0.8)).astype(int)
            X['low_credit_amount'] = (X['credit_amount'] < X['credit_amount'].quantile(0.2)).astype(int)
        
        return X
    
    def _final_cleaning(self, X: pd.DataFrame) -> pd.DataFrame:
        """Final cleaning of features"""
        logger.info("Performing final feature cleaning...")
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values with median
        X = X.fillna(X.median())
        
        # Ensure all features are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].median())
        
        # Remove constant features
        constant_features = [col for col in X.columns if X[col].std() == 0]
        if constant_features:
            logger.info(f"Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
        
        # Remove duplicate features
        duplicate_features = X.columns[X.columns.duplicated()].tolist()
        if duplicate_features:
            logger.info(f"Removing {len(duplicate_features)} duplicate features")
            X = X.loc[:, ~X.columns.duplicated()]
        
        logger.info(f"Final feature count: {len(X.columns)}")
        return X
    
    async def get_multiple_datasets(self, dataset_names: List[str] = None, 
                                  sample_size: Optional[int] = None) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Get multiple datasets for ensemble training"""
        if dataset_names is None:
            dataset_names = list(self.credit_datasets.keys())
        
        datasets = {}
        
        for dataset_name in dataset_names:
            try:
                logger.info(f"Loading dataset: {dataset_name}")
                X, y = await self.get_training_dataset(dataset_name, sample_size)
                
                if X is not None and len(X) > 0:
                    datasets[dataset_name] = (X, y)
                    logger.info(f"Successfully loaded {dataset_name}: {len(X)} samples")
                else:
                    logger.warning(f"Failed to load {dataset_name}")
                    
            except Exception as e:
                logger.error(f"Error loading {dataset_name}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(datasets)} datasets")
        return datasets
    
    def get_dataset_info(self, dataset_name: str = None) -> Dict[str, Any]:
        """Get information about available datasets"""
        if dataset_name:
            if dataset_name in self.credit_datasets:
                return self.credit_datasets[dataset_name]
            else:
                return None
        
        return self.credit_datasets
    
    def get_feature_summary(self, dataset_name: str) -> Dict[str, Any]:
        """Get feature summary for a specific dataset"""
        dataset_info = self.credit_datasets.get(dataset_name)
        if not dataset_info:
            return None
        
        return {
            'dataset_name': dataset_name,
            'description': dataset_info['description'],
            'feature_count': len(dataset_info['features']),
            'features': dataset_info['features'],
            'target': dataset_info['target'],
            'target_classes': list(dataset_info['target_mapping'].values()),
            'target_mapping': dataset_info['target_mapping']
        }
