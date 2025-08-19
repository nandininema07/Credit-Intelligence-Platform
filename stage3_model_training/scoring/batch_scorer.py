"""
Batch scoring operations for Stage 3.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
import joblib
import psutil

logger = logging.getLogger(__name__)

class BatchScorer:
    """Batch scoring engine for credit risk models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.batch_size = config.get('batch_size', 10000)
        self.n_workers = config.get('n_workers', psutil.cpu_count())
        self.use_multiprocessing = config.get('use_multiprocessing', True)
        
    def load_model(self, model_path: str):
        """Load trained model for scoring"""
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def set_model(self, model: Any):
        """Set model directly"""
        self.model = model
        
    async def score_batch(self, data: pd.DataFrame, 
                         output_path: str = None) -> Dict[str, Any]:
        """Score batch of data"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        start_time = datetime.now()
        
        # Split data into batches
        batches = self._create_batches(data)
        
        # Score batches
        if self.use_multiprocessing and len(batches) > 1:
            results = await self._score_parallel(batches)
        else:
            results = await self._score_sequential(batches)
        
        # Combine results
        combined_results = self._combine_batch_results(results)
        
        # Add metadata
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        final_results = {
            'scores': combined_results['scores'],
            'probabilities': combined_results['probabilities'],
            'metadata': {
                'n_samples': len(data),
                'n_batches': len(batches),
                'processing_time_seconds': processing_time,
                'samples_per_second': len(data) / processing_time,
                'timestamp': end_time.isoformat()
            }
        }
        
        # Save results if path provided
        if output_path:
            self._save_results(final_results, output_path)
        
        logger.info(f"Batch scoring completed: {len(data)} samples in {processing_time:.2f}s")
        return final_results
    
    def _create_batches(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Split data into batches"""
        batches = []
        for i in range(0, len(data), self.batch_size):
            batch = data.iloc[i:i + self.batch_size].copy()
            batches.append(batch)
        
        return batches
    
    async def _score_sequential(self, batches: List[pd.DataFrame]) -> List[Dict[str, Any]]:
        """Score batches sequentially"""
        results = []
        
        for i, batch in enumerate(batches):
            logger.debug(f"Processing batch {i+1}/{len(batches)}")
            result = self._score_single_batch(batch)
            results.append(result)
        
        return results
    
    async def _score_parallel(self, batches: List[pd.DataFrame]) -> List[Dict[str, Any]]:
        """Score batches in parallel"""
        
        if self.use_multiprocessing:
            # Use ProcessPoolExecutor for CPU-intensive tasks
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(executor, self._score_single_batch, batch)
                    for batch in batches
                ]
                results = await asyncio.gather(*tasks)
        else:
            # Use ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(executor, self._score_single_batch, batch)
                    for batch in batches
                ]
                results = await asyncio.gather(*tasks)
        
        return results
    
    def _score_single_batch(self, batch: pd.DataFrame) -> Dict[str, Any]:
        """Score a single batch"""
        try:
            # Get predictions
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(batch)[:, 1]
            else:
                probabilities = self.model.predict(batch)
            
            # Convert to credit scores (0-1000 scale)
            credit_scores = 1000 - (probabilities * 1000)
            credit_scores = credit_scores.astype(int)
            
            return {
                'scores': credit_scores,
                'probabilities': probabilities,
                'batch_size': len(batch),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error scoring batch: {e}")
            return {
                'scores': np.full(len(batch), -1),
                'probabilities': np.full(len(batch), -1),
                'batch_size': len(batch),
                'success': False,
                'error': str(e)
            }
    
    def _combine_batch_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple batches"""
        all_scores = []
        all_probabilities = []
        failed_batches = 0
        
        for result in results:
            if result['success']:
                all_scores.extend(result['scores'])
                all_probabilities.extend(result['probabilities'])
            else:
                failed_batches += 1
        
        return {
            'scores': np.array(all_scores),
            'probabilities': np.array(all_probabilities),
            'failed_batches': failed_batches,
            'total_batches': len(results)
        }
    
    def _save_results(self, results: Dict[str, Any], output_path: str):
        """Save scoring results"""
        
        # Create DataFrame with results
        results_df = pd.DataFrame({
            'credit_score': results['scores'],
            'default_probability': results['probabilities'],
            'scoring_timestamp': results['metadata']['timestamp']
        })
        
        # Save to file
        if output_path.endswith('.csv'):
            results_df.to_csv(output_path, index=False)
        elif output_path.endswith('.parquet'):
            results_df.to_parquet(output_path, index=False)
        else:
            # Default to pickle
            joblib.dump(results, output_path)
        
        logger.info(f"Results saved to {output_path}")
    
    async def score_streaming_data(self, data_stream, output_callback=None) -> Dict[str, Any]:
        """Score streaming data"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        total_processed = 0
        batch_buffer = []
        
        async for data_point in data_stream:
            batch_buffer.append(data_point)
            
            # Process when batch is full
            if len(batch_buffer) >= self.batch_size:
                batch_df = pd.DataFrame(batch_buffer)
                result = self._score_single_batch(batch_df)
                
                if output_callback:
                    await output_callback(result)
                
                total_processed += len(batch_buffer)
                batch_buffer = []
        
        # Process remaining data
        if batch_buffer:
            batch_df = pd.DataFrame(batch_buffer)
            result = self._score_single_batch(batch_df)
            
            if output_callback:
                await output_callback(result)
            
            total_processed += len(batch_buffer)
        
        return {'total_processed': total_processed}
    
    def get_scoring_statistics(self, scores: np.ndarray) -> Dict[str, Any]:
        """Get statistics about scoring results"""
        
        return {
            'total_samples': len(scores),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'percentiles': {
                '5th': float(np.percentile(scores, 5)),
                '25th': float(np.percentile(scores, 25)),
                '50th': float(np.percentile(scores, 50)),
                '75th': float(np.percentile(scores, 75)),
                '95th': float(np.percentile(scores, 95))
            },
            'score_distribution': {
                'excellent': int(np.sum(scores >= 800)),  # 800-1000
                'good': int(np.sum((scores >= 600) & (scores < 800))),  # 600-799
                'fair': int(np.sum((scores >= 400) & (scores < 600))),  # 400-599
                'poor': int(np.sum(scores < 400))  # 0-399
            }
        }

class DistributedBatchScorer:
    """Distributed batch scorer for large-scale scoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = None
        self.worker_configs = config.get('workers', [])
        
    def setup_distributed_scoring(self, model_path: str, worker_configs: List[Dict[str, Any]]):
        """Setup distributed scoring configuration"""
        self.model_path = model_path
        self.worker_configs = worker_configs
        
    async def score_distributed(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Score data using distributed workers"""
        
        if not self.worker_configs:
            # Fallback to local scoring
            local_scorer = BatchScorer(self.config)
            local_scorer.load_model(self.model_path)
            return await local_scorer.score_batch(data)
        
        # Split data across workers
        data_chunks = self._split_data_for_workers(data)
        
        # Submit scoring tasks to workers
        tasks = []
        for i, (worker_config, data_chunk) in enumerate(zip(self.worker_configs, data_chunks)):
            task = self._score_on_worker(worker_config, data_chunk, i)
            tasks.append(task)
        
        # Collect results
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        combined_results = self._combine_worker_results(worker_results)
        
        return combined_results
    
    def _split_data_for_workers(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Split data across available workers"""
        n_workers = len(self.worker_configs)
        chunk_size = len(data) // n_workers
        
        chunks = []
        for i in range(n_workers):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < n_workers - 1 else len(data)
            chunks.append(data.iloc[start_idx:end_idx].copy())
        
        return chunks
    
    async def _score_on_worker(self, worker_config: Dict[str, Any], 
                             data_chunk: pd.DataFrame, worker_id: int) -> Dict[str, Any]:
        """Score data chunk on a worker"""
        
        # This would typically involve sending data to a remote worker
        # For now, simulate with local processing
        
        try:
            local_scorer = BatchScorer(self.config)
            local_scorer.load_model(self.model_path)
            
            result = await local_scorer.score_batch(data_chunk)
            result['worker_id'] = worker_id
            result['worker_config'] = worker_config
            
            return result
            
        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {e}")
            return {
                'worker_id': worker_id,
                'error': str(e),
                'success': False
            }
    
    def _combine_worker_results(self, worker_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple workers"""
        
        all_scores = []
        all_probabilities = []
        failed_workers = 0
        total_processing_time = 0
        
        for result in worker_results:
            if isinstance(result, Exception):
                failed_workers += 1
                continue
                
            if result.get('success', True):
                all_scores.extend(result['scores'])
                all_probabilities.extend(result['probabilities'])
                total_processing_time += result['metadata']['processing_time_seconds']
            else:
                failed_workers += 1
        
        return {
            'scores': np.array(all_scores),
            'probabilities': np.array(all_probabilities),
            'metadata': {
                'total_samples': len(all_scores),
                'failed_workers': failed_workers,
                'total_workers': len(worker_results),
                'total_processing_time': total_processing_time,
                'timestamp': datetime.now().isoformat()
            }
        }
