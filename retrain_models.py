#!/usr/bin/env python3
"""
Model Retraining Script with Improved Feature Engineering
This script retrains the credit intelligence models using meaningful financial features
instead of generic feature_0 through feature_49.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import os # Added for os.getenv

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from stage3_model_training.main_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_retraining.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Main retraining function"""
    try:
        logger.info("Starting model retraining with improved feature engineering...")
        
        # Configuration for model training
        config = {
            'model_path': './models/',
            'feature_store_path': './feature_store/',
            'hyperparameter_tuning': True,
            'retrain_threshold_hours': 24,
            'retrain_threshold': 0.05,
            'model_types': ['xgboost', 'lightgbm', 'random_forest'],  # Specify which models to train
            
            # Real financial data configuration
            'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_API_KEY', ''),
            'news_api_key': os.getenv('NEWS_API_KEY', ''),
            'yahoo_finance_enabled': True,
            
            # Kaggle data configuration (PRIORITY)
            'kaggle_username': os.getenv('KAGGLE_USERNAME', ''),
            'kaggle_key': os.getenv('KAGGLE_API_KEY', ''),
            
            # Data quality settings
            'min_samples': 100,
            'max_features': 50,
            'validation_split': 0.3,
            'cross_validation_folds': 5
        }
        
        # Initialize model trainer
        trainer = ModelTrainer(config)
        await trainer.initialize()
        
        # Check if retraining is needed
        if await trainer.should_retrain():
            logger.info("Retraining is needed, proceeding...")
        else:
            logger.info("Models are up to date, but retraining anyway for improved features...")
        
        # Retrain all models using the correct method
        logger.info("Retraining all models...")
        results = await trainer.train_models()
        
        # Display results
        logger.info("\n" + "="*60)
        logger.info("MODEL RETRAINING RESULTS")
        logger.info("="*60)
        
        for model_type, result in results.items():
            if result.get('success'):
                perf = result.get('performance', {})
                logger.info(f"\n{model_type.upper()}:")
                logger.info(f"  Success: {result['success']}")
                logger.info(f"  Model Name: {result.get('model_name', 'Unknown')}")
                logger.info(f"  Accuracy: {perf.get('accuracy', 0):.3f}")
                logger.info(f"  Precision: {perf.get('precision', 0):.3f}")
                logger.info(f"  Recall: {perf.get('recall', 0):.3f}")
                logger.info(f"  F1-Score: {perf.get('f1_score', 0):.3f}")
                
                # Show cross-validation results
                if perf.get('cv_f1_mean') is not None:
                    logger.info(f"  CV F1-Score: {perf.get('cv_f1_mean', 0):.3f} ± {perf.get('cv_f1_std', 0):.3f}")
                
                # Show overfitting warnings
                if perf.get('overfitting_warning', False):
                    logger.warning(f"  OVERFITTING WARNING: Test F1 ({perf.get('f1_score', 0):.3f}) much higher than CV F1 ({perf.get('cv_f1_mean', 0):.3f})")
                    logger.warning(f"  Overfitting gap: {perf.get('overfitting_gap', 0):.3f}")
                
                logger.info(f"  Production Ready: {perf.get('validation_passed', False)}")
                
                if not perf.get('validation_passed', False):
                    logger.warning(f"  Warnings: {perf.get('validation_warnings', [])}")
            else:
                logger.error(f"\n{model_type.upper()}: Failed to train - {result.get('error', 'Unknown error')}")
        
        # Show production-ready models
        production_models = trainer.get_production_ready_models()
        fallback_models = trainer.get_fallback_models()
        
        logger.info("\n" + "="*60)
        logger.info("PRODUCTION STATUS")
        logger.info("="*60)
        
        if production_models:
            logger.info(f"Production Ready Models ({len(production_models)}):")
            for model in production_models:
                logger.info(f"  - {model}")
        else:
            logger.warning("No models meet production thresholds!")
        
        if fallback_models:
            logger.info(f"Fallback Models ({len(fallback_models)}):")
            for model in fallback_models:
                logger.info(f"  - {model}")
        
        # Recommendations
        logger.info("\n" + "="*60)
        logger.info("RECOMMENDATIONS")
        logger.info("="*60)
        
        if not production_models:
            logger.warning("CRITICAL: No models meet production thresholds!")
            logger.warning("   - Do not use these models for real credit decisions")
            logger.warning("   - Investigate feature engineering pipeline")
            logger.warning("   - Consider collecting more quality training data")
        else:
            logger.info("Models are ready for production use")
            logger.info("   - Monitor performance in production")
            logger.info("   - Set up automated retraining")
            logger.info("   - Implement A/B testing")
        
        logger.info("\nModel retraining completed!")
        
    except Exception as e:
        logger.error(f"Error during model retraining: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
