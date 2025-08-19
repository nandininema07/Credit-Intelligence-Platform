"""
Cross-sectional feature aggregation for Stage 2.
Handles aggregation across companies, sectors, and markets.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class CrossSectionalAggregator:
    """Cross-sectional feature aggregation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.aggregation_functions = {
            'mean': np.mean,
            'median': np.median,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'percentile_25': lambda x: np.percentile(x, 25),
            'percentile_75': lambda x: np.percentile(x, 75),
            'rank': self._calculate_rank,
            'zscore': self._calculate_zscore
        }
    
    async def aggregate_by_sector(self, data: pd.DataFrame, 
                                feature_columns: List[str],
                                sector_column: str = 'sector') -> pd.DataFrame:
        """Aggregate features by sector"""
        if data.empty or sector_column not in data.columns:
            return pd.DataFrame()
        
        sector_agg = []
        
        for sector in data[sector_column].unique():
            sector_data = data[data[sector_column] == sector]
            
            for feature_col in feature_columns:
                if feature_col not in sector_data.columns:
                    continue
                
                feature_values = sector_data[feature_col].dropna()
                if feature_values.empty:
                    continue
                
                # Calculate aggregations
                agg_result = {
                    'sector': sector,
                    'feature': feature_col,
                    'timestamp': datetime.now()
                }
                
                for agg_name, agg_func in self.aggregation_functions.items():
                    try:
                        if agg_name in ['rank', 'zscore']:
                            continue  # Skip relative measures for sector aggregation
                        
                        agg_value = agg_func(feature_values)
                        agg_result[f'{feature_col}_sector_{agg_name}'] = float(agg_value)
                    except Exception as e:
                        logger.warning(f"Error calculating {agg_name} for {feature_col}: {e}")
                
                sector_agg.append(agg_result)
        
        return pd.DataFrame(sector_agg)
    
    async def calculate_relative_features(self, data: pd.DataFrame,
                                        feature_columns: List[str],
                                        group_column: str = 'sector') -> pd.DataFrame:
        """Calculate relative features (ranks, z-scores)"""
        if data.empty:
            return data.copy()
        
        result_data = data.copy()
        
        for feature_col in feature_columns:
            if feature_col not in data.columns:
                continue
            
            # Calculate sector ranks
            result_data[f'{feature_col}_sector_rank'] = data.groupby(group_column)[feature_col].rank(
                method='dense', ascending=False, pct=True
            )
            
            # Calculate sector z-scores
            group_stats = data.groupby(group_column)[feature_col].agg(['mean', 'std'])
            
            def calculate_zscore(row):
                sector = row[group_column]
                value = row[feature_col]
                if pd.isna(value) or sector not in group_stats.index:
                    return np.nan
                
                sector_mean = group_stats.loc[sector, 'mean']
                sector_std = group_stats.loc[sector, 'std']
                
                if sector_std == 0:
                    return 0
                
                return (value - sector_mean) / sector_std
            
            result_data[f'{feature_col}_sector_zscore'] = data.apply(calculate_zscore, axis=1)
        
        return result_data
    
    def _calculate_rank(self, values: np.ndarray) -> float:
        """Calculate percentile rank"""
        return np.mean(values)  # Placeholder - actual rank needs group context
    
    def _calculate_zscore(self, values: np.ndarray) -> float:
        """Calculate z-score"""
        if len(values) < 2:
            return 0.0
        mean_val = np.mean(values)
        std_val = np.std(values)
        return (values[-1] - mean_val) / std_val if std_val > 0 else 0.0
