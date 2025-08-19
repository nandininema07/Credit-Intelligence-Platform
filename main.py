"""
Main entry point for the Credit Intelligence Platform.
Orchestrates the entire credit risk assessment pipeline.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List

from shared.utils.logger import setup_logger
from shared.utils.config_manager import ConfigManager
from stage1_data_ingestion.main_pipeline import DataIngestionPipeline
from stage2_feature_engineering.main_processor import FeatureProcessor
from stage3_model_training.training.train_pipeline import TrainingPipeline
from stage3_model_training.scoring.real_time_scorer import RealTimeScorer
from stage4_explainability.explainer.shap_explainer import SHAPExplainer
from stage4_explainability.chatbot.credit_chatbot import CreditChatbot
from stage5_alerting_workflows.alerting.alert_manager import AlertManager
from stage5_alerting_workflows.workflows.workflow_engine import WorkflowEngine

logger = setup_logger(__name__, log_file="./logs/main.log")

class CreditIntelligencePlatform:
    """Main Credit Intelligence Platform orchestrator"""
    
    def __init__(self, config_path: str = "./config"):
        self.config_manager = ConfigManager(config_path)
        self.components = {}
        
    async def initialize(self):
        """Initialize all platform components"""
        logger.info("Initializing Credit Intelligence Platform...")
        
        try:
            # Initialize core components
            config = self.config_manager.get('application', {})
            
            # Stage 1: Data Ingestion
            self.components['data_ingestion'] = DataIngestionPipeline(
                self.config_manager.get('database', {}),
                self.config_manager.get('aws', {}),
                self.config_manager.get_api_key
            )
            
            # Stage 2: Feature Engineering
            self.components['feature_processor'] = FeatureProcessor({})
            
            # Stage 3: Model Training & Scoring
            self.components['real_time_scorer'] = RealTimeScorer(
                self.config_manager.get('models', {})
            )
            
            # Stage 4: Explainability
            self.components['shap_explainer'] = SHAPExplainer({})
            self.components['chatbot'] = CreditChatbot({})
            
            # Stage 5: Alerting & Workflows
            self.components['alert_manager'] = AlertManager(
                self.config_manager.get('alerting', {})
            )
            self.components['workflow_engine'] = WorkflowEngine(
                self.config_manager.get('workflows', {})
            )
            
            # Load models
            await self.components['real_time_scorer'].load_models()
            
            logger.info("Platform initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Platform initialization failed: {str(e)}")
            raise
    
    async def process_company(self, company_id: str) -> Dict[str, Any]:
        """Process a single company through the entire pipeline"""
        logger.info(f"Processing company: {company_id}")
        
        try:
            # Stage 1: Data Ingestion
            raw_data = await self.components['data_ingestion'].ingest_company_data(company_id)
            
            # Stage 2: Feature Engineering
            features = await self.components['feature_processor'].process_company_data(
                company_id, raw_data
            )
            
            # Stage 3: Model Scoring
            from stage3_model_training.scoring.real_time_scorer import ScoringRequest
            scoring_request = ScoringRequest(
                company_id=company_id,
                features=features.features
            )
            
            scoring_result = await self.components['real_time_scorer'].score_single(scoring_request)
            
            # Stage 4: Generate Explanation
            # This would require model and background data setup
            
            # Stage 5: Check Alerts
            await self.components['alert_manager'].check_alerts(
                company_id, 
                {
                    'credit_score': scoring_result.credit_score,
                    'risk_category': scoring_result.risk_category,
                    'company_id': company_id
                }
            )
            
            result = {
                'company_id': company_id,
                'credit_score': scoring_result.credit_score,
                'risk_category': scoring_result.risk_category,
                'confidence': scoring_result.confidence_score,
                'features_processed': len(features.features),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Company {company_id} processed successfully. Score: {scoring_result.credit_score}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing company {company_id}: {str(e)}")
            raise
    
    async def batch_process(self, company_ids: List[str]) -> List[Dict[str, Any]]:
        """Process multiple companies"""
        logger.info(f"Starting batch processing for {len(company_ids)} companies")
        
        results = []
        for company_id in company_ids:
            try:
                result = await self.process_company(company_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {company_id}: {str(e)}")
                results.append({
                    'company_id': company_id,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        logger.info(f"Batch processing completed. {len(results)} results")
        return results
    
    def get_platform_status(self) -> Dict[str, Any]:
        """Get platform health status"""
        return {
            'status': 'running',
            'components': list(self.components.keys()),
            'timestamp': datetime.now().isoformat(),
            'version': self.config_manager.get('application.version', '1.0.0')
        }

async def main():
    """Main application entry point"""
    try:
        # Initialize platform
        platform = CreditIntelligencePlatform()
        await platform.initialize()
        
        # Example usage
        test_companies = ['AAPL', 'MSFT', 'GOOGL']
        results = await platform.batch_process(test_companies)
        
        print("Processing Results:")
        for result in results:
            print(f"Company: {result['company_id']}")
            if 'credit_score' in result:
                print(f"  Credit Score: {result['credit_score']}")
                print(f"  Risk Category: {result['risk_category']}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
            print()
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
