"""
Main pipeline runner for the Credit Intelligence Platform.
Orchestrates all 5 stages and provides a unified interface.
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import all stages
from stage1_data_ingestion.main_pipeline import DataIngestionPipeline
from stage2_feature_engineering.main_processor import MainProcessor
from stage3_model_training.main_trainer import ModelTrainer
from stage4_explainability.main_explainer import ExplainabilityEngine
from stage5_alerting_workflows.main_alerting import AlertingEngine

# Import shared utilities
from shared.utils.config_manager import ConfigManager
from shared.utils.logger_setup import setup_logging

logger = logging.getLogger(__name__)

class CreditIntelligencePipeline:
    """Main pipeline orchestrator for all 5 stages"""
    
    def __init__(self, config_path: str = "config/config.json"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Initialize all stages
        self.stage1_pipeline = None
        self.stage2_processor = None
        self.stage3_trainer = None
        self.stage4_explainer = None
        self.stage5_alerting = None
        
        self.running = False
        
    async def initialize(self):
        """Initialize all pipeline stages"""
        try:
            logger.info("Initializing Credit Intelligence Pipeline...")
            
            # Stage 1: Data Ingestion
            stage1_config = self.config.get('stage1', {})
            self.stage1_pipeline = DataIngestionPipeline(stage1_config)
            await self.stage1_pipeline.initialize()
            logger.info("✓ Stage 1 (Data Ingestion) initialized")
            
            # Stage 2: Feature Engineering
            stage2_config = self.config.get('stage2', {})
            self.stage2_processor = MainProcessor(stage2_config)
            await self.stage2_processor.initialize()
            logger.info("✓ Stage 2 (Feature Engineering) initialized")
            
            # Stage 3: Model Training
            stage3_config = self.config.get('stage3', {})
            self.stage3_trainer = ModelTrainer(stage3_config)
            await self.stage3_trainer.initialize()
            logger.info("✓ Stage 3 (Model Training) initialized")
            
            # Stage 4: Explainability
            stage4_config = self.config.get('stage4', {})
            self.stage4_explainer = ExplainabilityEngine(stage4_config)
            await self.stage4_explainer.initialize()
            logger.info("✓ Stage 4 (Explainability) initialized")
            
            # Stage 5: Alerting & Workflows
            stage5_config = self.config.get('stage5', {})
            self.stage5_alerting = AlertingEngine(stage5_config)
            await self.stage5_alerting.initialize()
            logger.info("✓ Stage 5 (Alerting & Workflows) initialized")
            
            logger.info("🚀 All pipeline stages initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    async def start_pipeline(self):
        """Start the complete pipeline"""
        if self.running:
            logger.warning("Pipeline is already running")
            return
            
        try:
            self.running = True
            logger.info("🚀 Starting Credit Intelligence Pipeline...")
            
            # Start all stages concurrently
            tasks = [
                asyncio.create_task(self.stage1_pipeline.start_pipeline(), name="Stage1"),
                asyncio.create_task(self._stage2_processing_loop(), name="Stage2"),
                asyncio.create_task(self._stage3_training_loop(), name="Stage3"),
                asyncio.create_task(self._stage4_explanation_loop(), name="Stage4"),
                asyncio.create_task(self.stage5_alerting.start_engine(), name="Stage5"),
                asyncio.create_task(self._pipeline_monitoring_loop(), name="Monitor")
            ]
            
            # Wait for all tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            await self.stop_pipeline()
    
    async def stop_pipeline(self):
        """Stop the complete pipeline"""
        self.running = False
        logger.info("🛑 Stopping Credit Intelligence Pipeline...")
        
        try:
            # Stop all stages
            if self.stage1_pipeline:
                await self.stage1_pipeline.stop_pipeline()
            if self.stage5_alerting:
                await self.stage5_alerting.stop_engine()
                
            logger.info("✓ Pipeline stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
    
    async def _stage2_processing_loop(self):
        """Stage 2 processing loop"""
        interval = self.config.get('stage2', {}).get('processing_interval', 600)  # 10 minutes
        
        while self.running:
            try:
                # Get new data from Stage 1
                companies = self.stage1_pipeline.company_registry.get_monitored_companies()
                
                for company in companies:
                    # Process features for each company
                    await self.stage2_processor.process_company_features(company.name)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Stage 2 processing: {e}")
                await asyncio.sleep(60)
    
    async def _stage3_training_loop(self):
        """Stage 3 model training loop"""
        interval = self.config.get('stage3', {}).get('training_interval', 3600)  # 1 hour
        
        while self.running:
            try:
                # Check if retraining is needed
                if await self.stage3_trainer.should_retrain():
                    await self.stage3_trainer.train_models()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Stage 3 training: {e}")
                await asyncio.sleep(300)
    
    async def _stage4_explanation_loop(self):
        """Stage 4 explanation generation loop"""
        interval = self.config.get('stage4', {}).get('explanation_interval', 1800)  # 30 minutes
        
        while self.running:
            try:
                # Generate explanations for recent score changes
                await self.stage4_explainer.generate_batch_explanations()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Stage 4 explanations: {e}")
                await asyncio.sleep(180)
    
    async def _pipeline_monitoring_loop(self):
        """Monitor overall pipeline health"""
        interval = 300  # 5 minutes
        
        while self.running:
            try:
                # Collect pipeline status from all stages
                status = await self.get_pipeline_status()
                
                # Log pipeline health
                if status['healthy']:
                    logger.info(f"Pipeline Status: ✓ Healthy - {status['summary']}")
                else:
                    logger.warning(f"Pipeline Status: ⚠️ Issues Detected - {status['issues']}")
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in pipeline monitoring: {e}")
                await asyncio.sleep(60)
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        try:
            status = {
                'running': self.running,
                'timestamp': datetime.now(),
                'stages': {}
            }
            
            # Stage 1 status
            if self.stage1_pipeline:
                status['stages']['stage1'] = await self.stage1_pipeline.get_pipeline_status()
            
            # Stage 2 status
            if self.stage2_processor:
                status['stages']['stage2'] = await self.stage2_processor.get_processing_status()
            
            # Stage 3 status
            if self.stage3_trainer:
                status['stages']['stage3'] = await self.stage3_trainer.get_training_status()
            
            # Stage 4 status
            if self.stage4_explainer:
                status['stages']['stage4'] = await self.stage4_explainer.get_explanation_status()
            
            # Stage 5 status
            if self.stage5_alerting:
                status['stages']['stage5'] = await self.stage5_alerting.get_alerting_status()
            
            # Overall health assessment
            healthy_stages = sum(1 for stage_status in status['stages'].values() 
                               if isinstance(stage_status, dict) and stage_status.get('healthy', False))
            total_stages = len(status['stages'])
            
            status['healthy'] = healthy_stages >= total_stages * 0.8  # 80% threshold
            status['health_score'] = healthy_stages / total_stages if total_stages > 0 else 0
            status['summary'] = f"{healthy_stages}/{total_stages} stages healthy"
            
            if not status['healthy']:
                issues = []
                for stage_name, stage_status in status['stages'].items():
                    if not stage_status.get('healthy', False):
                        issues.append(f"{stage_name}: {stage_status.get('error', 'Unknown issue')}")
                status['issues'] = issues
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {'running': False, 'healthy': False, 'error': str(e)}
    
    async def run_single_cycle(self, company_name: str) -> Dict[str, Any]:
        """Run a single processing cycle for testing"""
        try:
            logger.info(f"Running single cycle for {company_name}")
            
            # Stage 1: Ingest data
            await self.stage1_pipeline.run_single_ingestion(company_name)
            
            # Stage 2: Process features
            features = await self.stage2_processor.process_company_features(company_name)
            
            # Stage 3: Generate score
            score_result = await self.stage3_trainer.score_company(company_name)
            
            # Stage 4: Generate explanation
            explanation = await self.stage4_explainer.explain_score(company_name, score_result)
            
            # Stage 5: Check for alerts
            alerts = await self.stage5_alerting.check_company_alerts(company_name, score_result)
            
            result = {
                'company': company_name,
                'timestamp': datetime.now(),
                'features_count': len(features) if features else 0,
                'score': score_result.get('score') if score_result else None,
                'explanation': explanation,
                'alerts_generated': len(alerts) if alerts else 0,
                'success': True
            }
            
            logger.info(f"Single cycle completed for {company_name}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in single cycle for {company_name}: {e}")
            return {
                'company': company_name,
                'timestamp': datetime.now(),
                'success': False,
                'error': str(e)
            }
    
    async def cleanup(self):
        """Cleanup pipeline resources"""
        try:
            await self.stop_pipeline()
            
            if self.stage1_pipeline:
                await self.stage1_pipeline.cleanup()
            if self.stage2_processor:
                await self.stage2_processor.cleanup()
            if self.stage3_trainer:
                await self.stage3_trainer.cleanup()
            if self.stage4_explainer:
                await self.stage4_explainer.cleanup()
            if self.stage5_alerting:
                await self.stage5_alerting.cleanup()
                
            logger.info("✓ Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Credit Intelligence Pipeline')
    parser.add_argument('--config', default='config/config.json', help='Configuration file path')
    parser.add_argument('--mode', choices=['run', 'test', 'status'], default='run', help='Run mode')
    parser.add_argument('--company', help='Company name for test mode')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Initialize pipeline
    pipeline = CreditIntelligencePipeline(args.config)
    
    try:
        await pipeline.initialize()
        
        if args.mode == 'run':
            # Run continuous pipeline
            logger.info("Starting continuous pipeline...")
            await pipeline.start_pipeline()
            
        elif args.mode == 'test':
            # Run single test cycle
            if not args.company:
                logger.error("Company name required for test mode")
                return
                
            result = await pipeline.run_single_cycle(args.company)
            print(json.dumps(result, indent=2, default=str))
            
        elif args.mode == 'status':
            # Get pipeline status
            status = await pipeline.get_pipeline_status()
            print(json.dumps(status, indent=2, default=str))
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
