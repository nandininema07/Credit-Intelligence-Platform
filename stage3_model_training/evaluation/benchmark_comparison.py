"""
Benchmark comparison for Stage 3 model evaluation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

logger = logging.getLogger(__name__)

class BenchmarkComparator:
    """Compare models against standard benchmarks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.benchmarks = {}
        self.comparison_results = {}
        
    def create_baseline_benchmarks(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Create baseline benchmark models"""
        
        benchmarks = {}
        
        # Random classifier
        random_clf = DummyClassifier(strategy='uniform', random_state=42)
        random_clf.fit(X_train, y_train)
        benchmarks['random'] = random_clf
        
        # Majority class classifier
        majority_clf = DummyClassifier(strategy='most_frequent')
        majority_clf.fit(X_train, y_train)
        benchmarks['majority_class'] = majority_clf
        
        # Stratified classifier
        stratified_clf = DummyClassifier(strategy='stratified', random_state=42)
        stratified_clf.fit(X_train, y_train)
        benchmarks['stratified'] = stratified_clf
        
        # Simple logistic regression
        simple_lr = LogisticRegression(random_state=42, max_iter=1000)
        simple_lr.fit(X_train, y_train)
        benchmarks['simple_logistic'] = simple_lr
        
        # Simple random forest
        simple_rf = RandomForestClassifier(n_estimators=50, random_state=42)
        simple_rf.fit(X_train, y_train)
        benchmarks['simple_random_forest'] = simple_rf
        
        self.benchmarks = benchmarks
        
        # Evaluate benchmarks
        benchmark_results = {}
        for name, model in benchmarks.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            benchmark_results[name] = {
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan,
                'accuracy': accuracy_score(y_test, y_pred),
                'model': model
            }
        
        logger.info(f"Created {len(benchmarks)} baseline benchmarks")
        return benchmark_results
    
    def compare_with_benchmarks(self, target_model: Any, X_test: pd.DataFrame, 
                              y_test: pd.Series, model_name: str = "Target Model") -> Dict[str, Any]:
        """Compare target model with benchmarks"""
        
        if not self.benchmarks:
            raise ValueError("No benchmarks created. Call create_baseline_benchmarks first.")
        
        # Evaluate target model
        target_pred_proba = target_model.predict_proba(X_test)[:, 1]
        target_pred = target_model.predict(X_test)
        
        target_results = {
            'roc_auc': roc_auc_score(y_test, target_pred_proba) if len(np.unique(y_test)) > 1 else np.nan,
            'accuracy': accuracy_score(y_test, target_pred)
        }
        
        # Compare with each benchmark
        comparisons = {}
        
        for benchmark_name, benchmark_model in self.benchmarks.items():
            benchmark_pred_proba = benchmark_model.predict_proba(X_test)[:, 1]
            benchmark_pred = benchmark_model.predict(X_test)
            
            benchmark_results = {
                'roc_auc': roc_auc_score(y_test, benchmark_pred_proba) if len(np.unique(y_test)) > 1 else np.nan,
                'accuracy': accuracy_score(y_test, benchmark_pred)
            }
            
            # Calculate improvements
            auc_improvement = target_results['roc_auc'] - benchmark_results['roc_auc'] if not (np.isnan(target_results['roc_auc']) or np.isnan(benchmark_results['roc_auc'])) else np.nan
            accuracy_improvement = target_results['accuracy'] - benchmark_results['accuracy']
            
            comparisons[benchmark_name] = {
                'benchmark_auc': benchmark_results['roc_auc'],
                'benchmark_accuracy': benchmark_results['accuracy'],
                'auc_improvement': auc_improvement,
                'accuracy_improvement': accuracy_improvement,
                'auc_improvement_percentage': (auc_improvement / benchmark_results['roc_auc'] * 100) if not (np.isnan(auc_improvement) or benchmark_results['roc_auc'] == 0) else np.nan,
                'accuracy_improvement_percentage': (accuracy_improvement / benchmark_results['accuracy'] * 100) if benchmark_results['accuracy'] != 0 else np.nan
            }
        
        # Overall assessment
        assessment = self._assess_model_performance(target_results, comparisons)
        
        return {
            'model_name': model_name,
            'target_model_results': target_results,
            'benchmark_comparisons': comparisons,
            'assessment': assessment
        }
    
    def _assess_model_performance(self, target_results: Dict[str, Any], 
                                comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall model performance against benchmarks"""
        
        # Count improvements
        auc_improvements = [c['auc_improvement'] for c in comparisons.values() if not np.isnan(c['auc_improvement'])]
        accuracy_improvements = [c['accuracy_improvement'] for c in comparisons.values()]
        
        positive_auc_improvements = sum(1 for imp in auc_improvements if imp > 0)
        positive_accuracy_improvements = sum(1 for imp in accuracy_improvements if imp > 0)
        
        # Performance level
        target_auc = target_results['roc_auc']
        
        if not np.isnan(target_auc):
            if target_auc >= 0.8:
                performance_level = 'excellent'
            elif target_auc >= 0.7:
                performance_level = 'good'
            elif target_auc >= 0.6:
                performance_level = 'fair'
            else:
                performance_level = 'poor'
        else:
            performance_level = 'unknown'
        
        return {
            'performance_level': performance_level,
            'beats_benchmarks': {
                'auc': f"{positive_auc_improvements}/{len(auc_improvements)}" if auc_improvements else "0/0",
                'accuracy': f"{positive_accuracy_improvements}/{len(accuracy_improvements)}"
            },
            'mean_auc_improvement': np.mean(auc_improvements) if auc_improvements else np.nan,
            'mean_accuracy_improvement': np.mean(accuracy_improvements),
            'recommendation': self._get_performance_recommendation(performance_level, positive_auc_improvements, len(auc_improvements))
        }
    
    def _get_performance_recommendation(self, performance_level: str, 
                                      auc_improvements: int, total_benchmarks: int) -> str:
        """Get recommendation based on performance"""
        
        if performance_level == 'excellent' and auc_improvements == total_benchmarks:
            return "Model shows excellent performance and beats all benchmarks"
        elif performance_level in ['good', 'excellent'] and auc_improvements >= total_benchmarks * 0.8:
            return "Model shows strong performance and beats most benchmarks"
        elif performance_level == 'fair' or auc_improvements >= total_benchmarks * 0.5:
            return "Model shows acceptable performance but may need improvement"
        else:
            return "Model performance is below expectations - consider redesign"
    
    def industry_benchmark_comparison(self, model_results: Dict[str, Any],
                                    industry_benchmarks: Dict[str, float] = None) -> Dict[str, Any]:
        """Compare against industry benchmarks"""
        
        if industry_benchmarks is None:
            # Default industry benchmarks for credit risk
            industry_benchmarks = {
                'consumer_credit_auc': 0.75,
                'commercial_credit_auc': 0.70,
                'mortgage_auc': 0.80,
                'credit_card_auc': 0.72
            }
        
        model_auc = model_results.get('roc_auc', np.nan)
        
        comparisons = {}
        for benchmark_name, benchmark_value in industry_benchmarks.items():
            if not np.isnan(model_auc):
                difference = model_auc - benchmark_value
                percentage_diff = (difference / benchmark_value) * 100
                
                comparisons[benchmark_name] = {
                    'benchmark_value': benchmark_value,
                    'model_value': model_auc,
                    'difference': difference,
                    'percentage_difference': percentage_diff,
                    'beats_benchmark': difference > 0
                }
        
        return {
            'industry_comparisons': comparisons,
            'beats_industry_benchmarks': sum(1 for c in comparisons.values() if c['beats_benchmark']),
            'total_industry_benchmarks': len(comparisons)
        }
    
    def statistical_significance_test(self, model1_predictions: np.ndarray,
                                    model2_predictions: np.ndarray,
                                    y_true: pd.Series) -> Dict[str, Any]:
        """Test statistical significance of performance difference"""
        
        from scipy.stats import ttest_rel, wilcoxon
        
        # Calculate individual sample AUCs using bootstrap
        n_bootstrap = 1000
        model1_aucs = []
        model2_aucs = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(len(y_true), len(y_true), replace=True)
            y_bootstrap = y_true.iloc[bootstrap_indices]
            pred1_bootstrap = model1_predictions[bootstrap_indices]
            pred2_bootstrap = model2_predictions[bootstrap_indices]
            
            if len(np.unique(y_bootstrap)) > 1:
                auc1 = roc_auc_score(y_bootstrap, pred1_bootstrap)
                auc2 = roc_auc_score(y_bootstrap, pred2_bootstrap)
                model1_aucs.append(auc1)
                model2_aucs.append(auc2)
        
        if not model1_aucs:
            return {'status': 'insufficient_data'}
        
        # Paired t-test
        t_stat, t_p_value = ttest_rel(model1_aucs, model2_aucs)
        
        # Wilcoxon signed-rank test
        w_stat, w_p_value = wilcoxon(model1_aucs, model2_aucs)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(model1_aucs) + np.var(model2_aucs)) / 2)
        cohens_d = (np.mean(model1_aucs) - np.mean(model2_aucs)) / pooled_std if pooled_std > 0 else 0
        
        return {
            'model1_mean_auc': np.mean(model1_aucs),
            'model2_mean_auc': np.mean(model2_aucs),
            'auc_difference': np.mean(model1_aucs) - np.mean(model2_aucs),
            'paired_ttest': {
                'statistic': t_stat,
                'p_value': t_p_value,
                'significant': t_p_value < 0.05
            },
            'wilcoxon_test': {
                'statistic': w_stat,
                'p_value': w_p_value,
                'significant': w_p_value < 0.05
            },
            'effect_size': {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_effect_size(cohens_d)
            },
            'n_bootstrap_samples': len(model1_aucs)
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def generate_benchmark_report(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                                X_train: pd.DataFrame, y_train: pd.Series,
                                model_name: str = "Target Model") -> Dict[str, Any]:
        """Generate comprehensive benchmark comparison report"""
        
        # Create benchmarks
        benchmark_results = self.create_baseline_benchmarks(X_train, y_train, X_test, y_test)
        
        # Compare with benchmarks
        comparison_results = self.compare_with_benchmarks(model, X_test, y_test, model_name)
        
        # Industry comparison
        industry_comparison = self.industry_benchmark_comparison(comparison_results['target_model_results'])
        
        # Statistical significance tests
        target_predictions = model.predict_proba(X_test)[:, 1]
        significance_tests = {}
        
        for benchmark_name, benchmark_model in self.benchmarks.items():
            benchmark_predictions = benchmark_model.predict_proba(X_test)[:, 1]
            sig_test = self.statistical_significance_test(
                target_predictions, benchmark_predictions, y_test
            )
            significance_tests[benchmark_name] = sig_test
        
        return {
            'model_name': model_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'benchmark_results': benchmark_results,
            'comparison_results': comparison_results,
            'industry_comparison': industry_comparison,
            'significance_tests': significance_tests,
            'summary': self._generate_summary(comparison_results, industry_comparison)
        }
    
    def _generate_summary(self, comparison_results: Dict[str, Any],
                         industry_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of benchmark comparison"""
        
        assessment = comparison_results['assessment']
        
        summary = {
            'performance_level': assessment['performance_level'],
            'beats_baseline_benchmarks': assessment['beats_benchmarks'],
            'beats_industry_benchmarks': f"{industry_comparison['beats_industry_benchmarks']}/{industry_comparison['total_industry_benchmarks']}",
            'overall_recommendation': assessment['recommendation']
        }
        
        return summary
