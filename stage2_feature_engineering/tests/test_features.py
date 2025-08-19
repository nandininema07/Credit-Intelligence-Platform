"""
Tests for feature engineering components in Stage 2.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from ..feature_store.feature_store import FeatureStore
from ..feature_store.feature_registry import FeatureRegistry, FeatureDefinition, FeatureType, FeatureStatus
from ..feature_store.feature_validation import FeatureValidator, ValidationSeverity
from ..aggregation.time_series_agg import TimeSeriesAggregator
from ..aggregation.cross_sectional_agg import CrossSectionalAggregator
from ..aggregation.rolling_metrics import RollingMetricsCalculator
from ..transformers.scalers import FeatureScaler
from ..transformers.encoders import CategoricalEncoder
from ..transformers.feature_selection import FeatureSelector

@pytest.fixture
def sample_feature_data():
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    companies = ['AAPL', 'MSFT', 'GOOGL'] * 17  # 51 total, trim to 50
    companies = companies[:50]
    
    return pd.DataFrame({
        'timestamp': dates,
        'company': companies,
        'sector': ['Technology'] * 50,
        'sentiment_score': np.random.normal(0, 0.3, 50),
        'stock_price': 100 + np.cumsum(np.random.normal(0, 2, 50)),
        'volume': np.random.uniform(1000000, 5000000, 50),
        'revenue': np.random.uniform(80, 120, 50),
        'debt_ratio': np.random.uniform(0.2, 0.8, 50),
        'category_feature': np.random.choice(['A', 'B', 'C'], 50)
    })

@pytest.fixture
def sample_config():
    return {
        'feature_store': {
            'storage_backend': 'postgresql'
        },
        'validation': {
            'null_threshold': 10.0,
            'correlation_threshold': 0.95
        },
        'scaling': {
            'method': 'standard'
        }
    }

class TestFeatureStore:
    """Test feature store functionality"""
    
    def test_feature_store_init(self, sample_config):
        store = FeatureStore(sample_config)
        assert store.config == sample_config
    
    @pytest.mark.asyncio
    async def test_store_features(self, sample_config, sample_feature_data):
        store = FeatureStore(sample_config)
        
        # Mock storage backend
        with patch.object(store, '_store_to_backend', return_value=True):
            result = await store.store_features(sample_feature_data, 'test_features')
            assert result == True
    
    @pytest.mark.asyncio
    async def test_retrieve_features(self, sample_config):
        store = FeatureStore(sample_config)
        
        # Mock retrieval
        mock_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        with patch.object(store, '_retrieve_from_backend', return_value=mock_data):
            result = await store.retrieve_features('test_features')
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3

class TestFeatureRegistry:
    """Test feature registry functionality"""
    
    def test_feature_registry_init(self, sample_config):
        registry = FeatureRegistry(sample_config)
        assert registry.config == sample_config
        assert isinstance(registry.features, dict)
    
    def test_register_feature(self, sample_config):
        registry = FeatureRegistry(sample_config)
        
        feature_def = FeatureDefinition(
            name="test_feature",
            feature_type=FeatureType.NUMERICAL,
            description="Test feature",
            source_table="test_table",
            transformation="sum",
            dependencies=[],
            tags=["test"],
            owner="test_user",
            status=FeatureStatus.ACTIVE,
            created_date=datetime.now(),
            last_updated=datetime.now()
        )
        
        result = registry.register_feature(feature_def)
        assert result == True
        assert "test_feature" in registry.features
    
    def test_get_feature(self, sample_config):
        registry = FeatureRegistry(sample_config)
        
        # Register a feature first
        feature_def = FeatureDefinition(
            name="test_feature",
            feature_type=FeatureType.NUMERICAL,
            description="Test feature",
            source_table="test_table",
            transformation="sum",
            dependencies=[],
            tags=["test"],
            owner="test_user",
            status=FeatureStatus.ACTIVE,
            created_date=datetime.now(),
            last_updated=datetime.now()
        )
        registry.register_feature(feature_def)
        
        # Retrieve feature
        retrieved = registry.get_feature("test_feature")
        assert retrieved is not None
        assert retrieved.name == "test_feature"
    
    def test_list_features_with_filters(self, sample_config):
        registry = FeatureRegistry(sample_config)
        
        # Register multiple features
        for i in range(3):
            feature_def = FeatureDefinition(
                name=f"feature_{i}",
                feature_type=FeatureType.NUMERICAL if i % 2 == 0 else FeatureType.CATEGORICAL,
                description=f"Test feature {i}",
                source_table="test_table",
                transformation="sum",
                dependencies=[],
                tags=["test"],
                owner="test_user",
                status=FeatureStatus.ACTIVE if i < 2 else FeatureStatus.DEPRECATED,
                created_date=datetime.now(),
                last_updated=datetime.now()
            )
            registry.register_feature(feature_def)
        
        # Test filtering
        active_features = registry.list_features(status=FeatureStatus.ACTIVE)
        assert len(active_features) == 2
        
        numerical_features = registry.list_features(feature_type=FeatureType.NUMERICAL)
        assert len(numerical_features) == 2  # features 0 and 2

class TestFeatureValidator:
    """Test feature validation functionality"""
    
    def test_feature_validator_init(self, sample_config):
        validator = FeatureValidator(sample_config)
        assert validator.config == sample_config
    
    def test_null_check(self, sample_config):
        validator = FeatureValidator(sample_config)
        
        # Data with nulls
        data_with_nulls = pd.Series([1, 2, None, 4, None])
        result = validator.check_null_values("test_feature", data_with_nulls)
        
        assert result.feature_name == "test_feature"
        assert result.check_name == "null_check"
        assert result.details['null_percentage'] == 40.0  # 2 out of 5
    
    def test_range_check(self, sample_config):
        validator = FeatureValidator(sample_config)
        
        # Normal data
        normal_data = pd.Series([1, 2, 3, 4, 5])
        result = validator.check_value_range("test_feature", normal_data)
        
        assert result.feature_name == "test_feature"
        assert result.check_name == "range_check"
        assert isinstance(result.details['min_value'], float)
        assert isinstance(result.details['max_value'], float)
    
    def test_distribution_check(self, sample_config):
        validator = FeatureValidator(sample_config)
        
        # Normally distributed data
        normal_data = pd.Series(np.random.normal(0, 1, 100))
        result = validator.check_distribution("test_feature", normal_data)
        
        assert result.feature_name == "test_feature"
        assert result.check_name == "distribution_check"
        assert 'mean' in result.details
        assert 'std' in result.details
    
    def test_validate_dataset(self, sample_config, sample_feature_data):
        validator = FeatureValidator(sample_config)
        
        # Select numeric columns for validation
        numeric_data = sample_feature_data.select_dtypes(include=[np.number])
        results = validator.validate_dataset(numeric_data)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Each column should have validation results
        for column in numeric_data.columns:
            assert column in results
            assert isinstance(results[column], list)

class TestTimeSeriesAggregator:
    """Test time series aggregation"""
    
    def test_time_series_aggregator_init(self, sample_config):
        aggregator = TimeSeriesAggregator(sample_config)
        assert aggregator.config == sample_config
    
    @pytest.mark.asyncio
    async def test_aggregate_features(self, sample_config, sample_feature_data):
        aggregator = TimeSeriesAggregator(sample_config)
        
        feature_columns = ['sentiment_score', 'stock_price', 'volume']
        result = await aggregator.aggregate_features(
            sample_feature_data, 
            feature_columns,
            'timestamp',
            'company'
        )
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert 'company' in result.columns
            assert 'window' in result.columns

class TestCrossSectionalAggregator:
    """Test cross-sectional aggregation"""
    
    @pytest.mark.asyncio
    async def test_aggregate_by_sector(self, sample_config, sample_feature_data):
        aggregator = CrossSectionalAggregator(sample_config)
        
        feature_columns = ['sentiment_score', 'stock_price']
        result = await aggregator.aggregate_by_sector(
            sample_feature_data,
            feature_columns,
            'sector'
        )
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert 'sector' in result.columns
            assert 'feature' in result.columns

class TestFeatureScaler:
    """Test feature scaling"""
    
    def test_feature_scaler_init(self, sample_config):
        scaler = FeatureScaler(sample_config)
        assert scaler.config == sample_config
    
    def test_fit_transform(self, sample_config, sample_feature_data):
        scaler = FeatureScaler(sample_config)
        
        numeric_data = sample_feature_data.select_dtypes(include=[np.number])
        scaled_data = scaler.fit_transform(numeric_data)
        
        assert isinstance(scaled_data, pd.DataFrame)
        assert scaled_data.shape == numeric_data.shape
        assert scaler.fitted == True

class TestCategoricalEncoder:
    """Test categorical encoding"""
    
    def test_categorical_encoder_init(self, sample_config):
        encoder = CategoricalEncoder(sample_config)
        assert encoder.config == sample_config
    
    def test_fit_transform(self, sample_config, sample_feature_data):
        encoder = CategoricalEncoder(sample_config)
        
        # Use only categorical columns
        categorical_data = sample_feature_data[['category_feature']]
        encoded_data = encoder.fit_transform(categorical_data)
        
        assert isinstance(encoded_data, pd.DataFrame)
        assert encoder.fitted == True

class TestFeatureSelector:
    """Test feature selection"""
    
    def test_feature_selector_init(self, sample_config):
        selector = FeatureSelector(sample_config)
        assert selector.config == sample_config
    
    def test_fit(self, sample_config, sample_feature_data):
        selector = FeatureSelector(sample_config)
        
        # Create target variable
        target = pd.Series(np.random.choice([0, 1], len(sample_feature_data)))
        
        # Select numeric features
        X = sample_feature_data.select_dtypes(include=[np.number])
        
        selector.fit(X, target)
        assert selector.fitted == True
        assert len(selector.selected_features) > 0

@pytest.mark.integration
class TestFeatureEngineeeringIntegration:
    """Test integrated feature engineering pipeline"""
    
    @pytest.mark.asyncio
    async def test_full_feature_pipeline(self, sample_config, sample_feature_data):
        """Test complete feature engineering pipeline"""
        
        # Initialize components
        registry = FeatureRegistry(sample_config)
        validator = FeatureValidator(sample_config)
        scaler = FeatureScaler(sample_config)
        
        # Register some features
        for column in sample_feature_data.select_dtypes(include=[np.number]).columns:
            if column != 'timestamp':
                feature_def = FeatureDefinition(
                    name=column,
                    feature_type=FeatureType.NUMERICAL,
                    description=f"Feature {column}",
                    source_table="raw_data",
                    transformation="direct",
                    dependencies=[],
                    tags=["test"],
                    owner="test_user",
                    status=FeatureStatus.ACTIVE,
                    created_date=datetime.now(),
                    last_updated=datetime.now()
                )
                registry.register_feature(feature_def)
        
        # Validate features
        numeric_data = sample_feature_data.select_dtypes(include=[np.number])
        validation_results = validator.validate_dataset(numeric_data)
        
        # Scale features
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Verify pipeline results
        assert len(registry.features) > 0
        assert len(validation_results) > 0
        assert isinstance(scaled_data, pd.DataFrame)
        assert scaled_data.shape == numeric_data.shape
    
    def test_feature_pipeline_error_handling(self, sample_config):
        """Test error handling in feature pipeline"""
        
        # Test with empty data
        empty_data = pd.DataFrame()
        
        scaler = FeatureScaler(sample_config)
        scaled = scaler.fit_transform(empty_data)
        assert scaled.empty
        
        validator = FeatureValidator(sample_config)
        validation_results = validator.validate_dataset(empty_data)
        assert len(validation_results) == 0
    
    def test_feature_consistency(self, sample_config, sample_feature_data):
        """Test feature consistency across transformations"""
        
        # Original data
        original_shape = sample_feature_data.shape
        
        # Apply transformations
        scaler = FeatureScaler(sample_config)
        numeric_data = sample_feature_data.select_dtypes(include=[np.number])
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Check consistency
        assert scaled_data.shape[0] == numeric_data.shape[0]  # Same number of rows
        assert not scaled_data.isnull().all().any()  # No completely null columns
    
    @pytest.mark.asyncio
    async def test_aggregation_consistency(self, sample_config, sample_feature_data):
        """Test aggregation consistency"""
        
        ts_aggregator = TimeSeriesAggregator(sample_config)
        cs_aggregator = CrossSectionalAggregator(sample_config)
        
        feature_columns = ['sentiment_score', 'stock_price']
        
        # Time series aggregation
        ts_result = await ts_aggregator.aggregate_features(
            sample_feature_data, feature_columns
        )
        
        # Cross-sectional aggregation
        cs_result = await cs_aggregator.aggregate_by_sector(
            sample_feature_data, feature_columns
        )
        
        # Both should return DataFrames
        assert isinstance(ts_result, pd.DataFrame)
        assert isinstance(cs_result, pd.DataFrame)
    
    def test_feature_metadata_consistency(self, sample_config):
        """Test feature metadata consistency"""
        
        registry = FeatureRegistry(sample_config)
        
        # Register feature
        feature_def = FeatureDefinition(
            name="test_consistency",
            feature_type=FeatureType.NUMERICAL,
            description="Consistency test feature",
            source_table="test_table",
            transformation="mean",
            dependencies=["base_feature"],
            tags=["test", "consistency"],
            owner="test_user",
            status=FeatureStatus.ACTIVE,
            created_date=datetime.now(),
            last_updated=datetime.now()
        )
        
        registry.register_feature(feature_def)
        
        # Retrieve and verify
        retrieved = registry.get_feature("test_consistency")
        assert retrieved.name == feature_def.name
        assert retrieved.feature_type == feature_def.feature_type
        assert retrieved.dependencies == feature_def.dependencies
    
    def test_feature_selection_consistency(self, sample_config, sample_feature_data):
        """Test feature selection consistency"""
        
        selector = FeatureSelector(sample_config)
        
        # Create binary target
        target = pd.Series(np.random.choice([0, 1], len(sample_feature_data)))
        X = sample_feature_data.select_dtypes(include=[np.number])
        
        # Fit selector
        selector.fit(X, target)
        
        # Get selected features from different methods
        ensemble_features = selector.get_selected_features('ensemble')
        
        # Verify consistency
        assert isinstance(ensemble_features, list)
        assert all(feature in X.columns for feature in ensemble_features)
        
        # Feature importance should be available
        importance = selector.get_feature_importance('ensemble')
        assert isinstance(importance, dict)
        assert len(importance) > 0
