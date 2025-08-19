"""
Counterfactual analysis for Stage 4.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class CounterfactualAnalyzer:
    """Generate and analyze counterfactual explanations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_names = None
        self.feature_ranges = None
        self.categorical_features = None
        self.scaler = None
        
    def initialize(self, model: Any, training_data: pd.DataFrame,
                  categorical_features: List[str] = None):
        """Initialize counterfactual analyzer"""
        
        self.model = model
        self.feature_names = list(training_data.columns)
        self.categorical_features = categorical_features or []
        
        # Calculate feature ranges for continuous features
        self.feature_ranges = {}
        for feature in self.feature_names:
            if feature not in self.categorical_features:
                self.feature_ranges[feature] = {
                    'min': float(training_data[feature].min()),
                    'max': float(training_data[feature].max()),
                    'mean': float(training_data[feature].mean()),
                    'std': float(training_data[feature].std())
                }
        
        # Initialize scaler for distance calculations
        self.scaler = StandardScaler()
        self.scaler.fit(training_data)
        
        logger.info("Counterfactual analyzer initialized")
    
    def generate_counterfactual(self, instance: pd.DataFrame,
                              desired_outcome: float = None,
                              max_changes: int = 3,
                              method: str = 'genetic') -> Dict[str, Any]:
        """Generate counterfactual explanation"""
        
        if self.model is None:
            raise ValueError("Analyzer not initialized. Call initialize first.")
        
        # Get current prediction
        if hasattr(self.model, 'predict_proba'):
            current_pred = self.model.predict_proba(instance)[0, 1]
        else:
            current_pred = self.model.predict(instance)[0]
        
        # Set desired outcome if not provided
        if desired_outcome is None:
            desired_outcome = 0.2 if current_pred > 0.5 else 0.8
        
        if method == 'genetic':
            return self._genetic_counterfactual(instance, desired_outcome, max_changes)
        elif method == 'gradient':
            return self._gradient_counterfactual(instance, desired_outcome, max_changes)
        elif method == 'nearest_neighbor':
            return self._nearest_neighbor_counterfactual(instance, desired_outcome)
        else:
            return self._simple_counterfactual(instance, desired_outcome, max_changes)
    
    def _simple_counterfactual(self, instance: pd.DataFrame,
                             desired_outcome: float,
                             max_changes: int) -> Dict[str, Any]:
        """Generate counterfactual using simple feature perturbation"""
        
        try:
            original_instance = instance.copy()
            current_instance = instance.copy()
            
            # Get current prediction
            if hasattr(self.model, 'predict_proba'):
                current_pred = self.model.predict_proba(current_instance)[0, 1]
            else:
                current_pred = self.model.predict(current_instance)[0]
            
            changes_made = {}
            iterations = 0
            max_iterations = 100
            
            # Calculate feature importance to guide changes
            feature_importance = self._calculate_feature_sensitivity(instance)
            
            # Sort features by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            for feature_name, importance in sorted_features:
                if len(changes_made) >= max_changes:
                    break
                
                if feature_name in self.categorical_features:
                    continue  # Skip categorical features for simplicity
                
                # Try different perturbations
                original_value = float(original_instance[feature_name].iloc[0])
                
                # Calculate perturbation direction and magnitude
                if (importance > 0 and desired_outcome < current_pred) or \
                   (importance < 0 and desired_outcome > current_pred):
                    # Decrease feature value
                    perturbations = [0.8, 0.6, 0.4, 0.2]
                else:
                    # Increase feature value
                    perturbations = [1.2, 1.5, 2.0, 3.0]
                
                for multiplier in perturbations:
                    new_value = original_value * multiplier
                    
                    # Check if within valid range
                    if feature_name in self.feature_ranges:
                        min_val = self.feature_ranges[feature_name]['min']
                        max_val = self.feature_ranges[feature_name]['max']
                        new_value = max(min_val, min(max_val, new_value))
                    
                    # Apply change
                    current_instance[feature_name] = new_value
                    
                    # Check new prediction
                    if hasattr(self.model, 'predict_proba'):
                        new_pred = self.model.predict_proba(current_instance)[0, 1]
                    else:
                        new_pred = self.model.predict(current_instance)[0]
                    
                    # Check if we've reached desired outcome
                    if abs(new_pred - desired_outcome) < abs(current_pred - desired_outcome):
                        changes_made[feature_name] = {
                            'original_value': original_value,
                            'new_value': float(new_value),
                            'change': float(new_value - original_value),
                            'relative_change': float((new_value - original_value) / original_value) if original_value != 0 else float('inf')
                        }
                        current_pred = new_pred
                        break
                    else:
                        # Revert change
                        current_instance[feature_name] = original_value
                
                iterations += 1
                if iterations >= max_iterations:
                    break
            
            # Calculate final metrics
            final_pred = current_pred
            distance = self._calculate_distance(original_instance, current_instance)
            
            return {
                'method': 'simple_counterfactual',
                'original_prediction': float(self.model.predict_proba(original_instance)[0, 1]) if hasattr(self.model, 'predict_proba') else float(self.model.predict(original_instance)[0]),
                'counterfactual_prediction': float(final_pred),
                'desired_outcome': float(desired_outcome),
                'success': abs(final_pred - desired_outcome) < 0.1,
                'changes_made': changes_made,
                'num_changes': len(changes_made),
                'distance': float(distance),
                'counterfactual_instance': current_instance.iloc[0].to_dict(),
                'generation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating simple counterfactual: {e}")
            return {'error': str(e)}
    
    def _genetic_counterfactual(self, instance: pd.DataFrame,
                              desired_outcome: float,
                              max_changes: int) -> Dict[str, Any]:
        """Generate counterfactual using genetic algorithm"""
        
        try:
            population_size = 50
            generations = 20
            mutation_rate = 0.1
            
            # Initialize population
            population = []
            original_values = instance.iloc[0].values
            
            for _ in range(population_size):
                individual = original_values.copy()
                
                # Randomly mutate some features
                num_mutations = np.random.randint(1, max_changes + 1)
                features_to_mutate = np.random.choice(
                    len(self.feature_names), num_mutations, replace=False
                )
                
                for feat_idx in features_to_mutate:
                    feature_name = self.feature_names[feat_idx]
                    
                    if feature_name not in self.categorical_features:
                        # Continuous feature
                        if feature_name in self.feature_ranges:
                            min_val = self.feature_ranges[feature_name]['min']
                            max_val = self.feature_ranges[feature_name]['max']
                            individual[feat_idx] = np.random.uniform(min_val, max_val)
                
                population.append(individual)
            
            best_individual = None
            best_fitness = float('inf')
            
            # Evolution loop
            for generation in range(generations):
                # Evaluate fitness
                fitness_scores = []
                
                for individual in population:
                    fitness = self._evaluate_fitness(
                        individual, desired_outcome, original_values
                    )
                    fitness_scores.append(fitness)
                    
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_individual = individual.copy()
                
                # Selection and reproduction
                new_population = []
                
                # Keep best individuals (elitism)
                elite_size = population_size // 10
                elite_indices = np.argsort(fitness_scores)[:elite_size]
                for idx in elite_indices:
                    new_population.append(population[idx].copy())
                
                # Generate offspring
                while len(new_population) < population_size:
                    # Tournament selection
                    parent1 = self._tournament_selection(population, fitness_scores)
                    parent2 = self._tournament_selection(population, fitness_scores)
                    
                    # Crossover
                    child = self._crossover(parent1, parent2)
                    
                    # Mutation
                    if np.random.random() < mutation_rate:
                        child = self._mutate(child, max_changes)
                    
                    new_population.append(child)
                
                population = new_population
            
            # Format result
            if best_individual is not None:
                counterfactual_df = pd.DataFrame(
                    [best_individual], columns=self.feature_names
                )
                
                changes_made = {}
                for i, feature_name in enumerate(self.feature_names):
                    if abs(best_individual[i] - original_values[i]) > 1e-6:
                        changes_made[feature_name] = {
                            'original_value': float(original_values[i]),
                            'new_value': float(best_individual[i]),
                            'change': float(best_individual[i] - original_values[i]),
                            'relative_change': float((best_individual[i] - original_values[i]) / original_values[i]) if original_values[i] != 0 else float('inf')
                        }
                
                final_pred = self.model.predict_proba(counterfactual_df)[0, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(counterfactual_df)[0]
                distance = self._calculate_distance(instance, counterfactual_df)
                
                return {
                    'method': 'genetic_counterfactual',
                    'original_prediction': float(self.model.predict_proba(instance)[0, 1]) if hasattr(self.model, 'predict_proba') else float(self.model.predict(instance)[0]),
                    'counterfactual_prediction': float(final_pred),
                    'desired_outcome': float(desired_outcome),
                    'success': abs(final_pred - desired_outcome) < 0.1,
                    'changes_made': changes_made,
                    'num_changes': len(changes_made),
                    'distance': float(distance),
                    'fitness': float(best_fitness),
                    'generations': generations,
                    'counterfactual_instance': counterfactual_df.iloc[0].to_dict(),
                    'generation_timestamp': datetime.now().isoformat()
                }
            else:
                return {'error': 'Failed to generate counterfactual'}
                
        except Exception as e:
            logger.error(f"Error generating genetic counterfactual: {e}")
            return {'error': str(e)}
    
    def _evaluate_fitness(self, individual: np.ndarray,
                         desired_outcome: float,
                         original_values: np.ndarray) -> float:
        """Evaluate fitness of an individual"""
        
        try:
            # Create DataFrame for prediction
            individual_df = pd.DataFrame([individual], columns=self.feature_names)
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                pred = self.model.predict_proba(individual_df)[0, 1]
            else:
                pred = self.model.predict(individual_df)[0]
            
            # Calculate fitness (lower is better)
            prediction_error = abs(pred - desired_outcome)
            distance_penalty = np.sum((individual - original_values) ** 2)
            
            fitness = prediction_error + 0.1 * distance_penalty
            
            return fitness
            
        except Exception:
            return float('inf')
    
    def _tournament_selection(self, population: List[np.ndarray],
                            fitness_scores: List[float],
                            tournament_size: int = 3) -> np.ndarray:
        """Tournament selection for genetic algorithm"""
        
        tournament_indices = np.random.choice(
            len(population), tournament_size, replace=False
        )
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Single-point crossover"""
        
        crossover_point = np.random.randint(1, len(parent1))
        child = np.concatenate([
            parent1[:crossover_point],
            parent2[crossover_point:]
        ])
        
        return child
    
    def _mutate(self, individual: np.ndarray, max_changes: int) -> np.ndarray:
        """Mutate individual"""
        
        mutated = individual.copy()
        
        # Select features to mutate
        num_mutations = np.random.randint(1, min(max_changes, len(individual)) + 1)
        features_to_mutate = np.random.choice(
            len(individual), num_mutations, replace=False
        )
        
        for feat_idx in features_to_mutate:
            feature_name = self.feature_names[feat_idx]
            
            if feature_name not in self.categorical_features:
                if feature_name in self.feature_ranges:
                    min_val = self.feature_ranges[feature_name]['min']
                    max_val = self.feature_ranges[feature_name]['max']
                    mutated[feat_idx] = np.random.uniform(min_val, max_val)
        
        return mutated
    
    def _gradient_counterfactual(self, instance: pd.DataFrame,
                               desired_outcome: float,
                               max_changes: int) -> Dict[str, Any]:
        """Generate counterfactual using gradient-based optimization"""
        
        try:
            current_instance = instance.copy()
            original_values = instance.iloc[0].values
            
            learning_rate = 0.01
            max_iterations = 100
            changes_made = {}
            
            for iteration in range(max_iterations):
                # Calculate gradients
                gradients = self._calculate_numerical_gradients(current_instance)
                
                # Get current prediction
                if hasattr(self.model, 'predict_proba'):
                    current_pred = self.model.predict_proba(current_instance)[0, 1]
                else:
                    current_pred = self.model.predict(current_instance)[0]
                
                # Check if we've reached desired outcome
                if abs(current_pred - desired_outcome) < 0.05:
                    break
                
                # Update features based on gradients
                prediction_error = desired_outcome - current_pred
                
                for i, feature_name in enumerate(self.feature_names):
                    if feature_name in self.categorical_features:
                        continue
                    
                    # Update feature value
                    gradient = gradients[i]
                    update = learning_rate * prediction_error * gradient
                    
                    new_value = current_instance[feature_name].iloc[0] + update
                    
                    # Clip to valid range
                    if feature_name in self.feature_ranges:
                        min_val = self.feature_ranges[feature_name]['min']
                        max_val = self.feature_ranges[feature_name]['max']
                        new_value = max(min_val, min(max_val, new_value))
                    
                    current_instance[feature_name] = new_value
            
            # Calculate changes
            for i, feature_name in enumerate(self.feature_names):
                current_value = current_instance[feature_name].iloc[0]
                original_value = original_values[i]
                
                if abs(current_value - original_value) > 1e-6:
                    changes_made[feature_name] = {
                        'original_value': float(original_value),
                        'new_value': float(current_value),
                        'change': float(current_value - original_value),
                        'relative_change': float((current_value - original_value) / original_value) if original_value != 0 else float('inf')
                    }
            
            final_pred = self.model.predict_proba(current_instance)[0, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(current_instance)[0]
            distance = self._calculate_distance(instance, current_instance)
            
            return {
                'method': 'gradient_counterfactual',
                'original_prediction': float(self.model.predict_proba(instance)[0, 1]) if hasattr(self.model, 'predict_proba') else float(self.model.predict(instance)[0]),
                'counterfactual_prediction': float(final_pred),
                'desired_outcome': float(desired_outcome),
                'success': abs(final_pred - desired_outcome) < 0.1,
                'changes_made': changes_made,
                'num_changes': len(changes_made),
                'distance': float(distance),
                'iterations': iteration + 1,
                'counterfactual_instance': current_instance.iloc[0].to_dict(),
                'generation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating gradient counterfactual: {e}")
            return {'error': str(e)}
    
    def _nearest_neighbor_counterfactual(self, instance: pd.DataFrame,
                                       desired_outcome: float) -> Dict[str, Any]:
        """Find counterfactual using nearest neighbor approach"""
        
        # This would require access to training data with different outcomes
        # For now, return a placeholder
        return {
            'method': 'nearest_neighbor_counterfactual',
            'error': 'Nearest neighbor method requires training data with diverse outcomes'
        }
    
    def _calculate_feature_sensitivity(self, instance: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature sensitivity for the instance"""
        
        sensitivity = {}
        epsilon = 1e-4
        
        # Get baseline prediction
        if hasattr(self.model, 'predict_proba'):
            baseline_pred = self.model.predict_proba(instance)[0, 1]
        else:
            baseline_pred = self.model.predict(instance)[0]
        
        for feature_name in self.feature_names:
            if feature_name in self.categorical_features:
                sensitivity[feature_name] = 0.0
                continue
            
            # Perturb feature
            perturbed_instance = instance.copy()
            original_value = instance[feature_name].iloc[0]
            perturbed_instance[feature_name] = original_value + epsilon
            
            try:
                if hasattr(self.model, 'predict_proba'):
                    perturbed_pred = self.model.predict_proba(perturbed_instance)[0, 1]
                else:
                    perturbed_pred = self.model.predict(perturbed_instance)[0]
                
                sensitivity[feature_name] = (perturbed_pred - baseline_pred) / epsilon
                
            except Exception:
                sensitivity[feature_name] = 0.0
        
        return sensitivity
    
    def _calculate_numerical_gradients(self, instance: pd.DataFrame,
                                     epsilon: float = 1e-4) -> np.ndarray:
        """Calculate numerical gradients"""
        
        gradients = np.zeros(len(self.feature_names))
        
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in self.categorical_features:
                gradients[i] = 0.0
                continue
            
            # Forward difference
            instance_plus = instance.copy()
            instance_plus[feature_name] += epsilon
            
            try:
                if hasattr(self.model, 'predict_proba'):
                    pred_plus = self.model.predict_proba(instance_plus)[0, 1]
                    pred_current = self.model.predict_proba(instance)[0, 1]
                else:
                    pred_plus = self.model.predict(instance_plus)[0]
                    pred_current = self.model.predict(instance)[0]
                
                gradients[i] = (pred_plus - pred_current) / epsilon
                
            except Exception:
                gradients[i] = 0.0
        
        return gradients
    
    def _calculate_distance(self, instance1: pd.DataFrame,
                          instance2: pd.DataFrame) -> float:
        """Calculate distance between two instances"""
        
        if self.scaler is not None:
            scaled1 = self.scaler.transform(instance1)
            scaled2 = self.scaler.transform(instance2)
            return float(pairwise_distances(scaled1, scaled2)[0, 0])
        else:
            return float(pairwise_distances(instance1.values, instance2.values)[0, 0])
    
    def analyze_counterfactual_diversity(self, instance: pd.DataFrame,
                                       num_counterfactuals: int = 5) -> Dict[str, Any]:
        """Generate diverse counterfactual explanations"""
        
        counterfactuals = []
        methods = ['simple', 'genetic']
        
        for i in range(num_counterfactuals):
            method = methods[i % len(methods)]
            
            # Vary desired outcome slightly
            base_pred = self.model.predict_proba(instance)[0, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(instance)[0]
            desired_outcome = 0.2 + (i * 0.15) if base_pred > 0.5 else 0.6 + (i * 0.1)
            desired_outcome = max(0.1, min(0.9, desired_outcome))
            
            counterfactual = self.generate_counterfactual(
                instance, desired_outcome=desired_outcome, method=method
            )
            
            if 'error' not in counterfactual:
                counterfactuals.append(counterfactual)
        
        # Analyze diversity
        if len(counterfactuals) > 1:
            diversity_metrics = self._calculate_diversity_metrics(counterfactuals)
        else:
            diversity_metrics = {'error': 'Insufficient counterfactuals generated'}
        
        return {
            'counterfactuals': counterfactuals,
            'diversity_metrics': diversity_metrics,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_diversity_metrics(self, counterfactuals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate diversity metrics for counterfactuals"""
        
        try:
            # Extract feature changes
            all_changes = []
            for cf in counterfactuals:
                changes = list(cf.get('changes_made', {}).keys())
                all_changes.append(set(changes))
            
            # Calculate Jaccard diversity
            jaccard_similarities = []
            for i in range(len(all_changes)):
                for j in range(i + 1, len(all_changes)):
                    intersection = len(all_changes[i] & all_changes[j])
                    union = len(all_changes[i] | all_changes[j])
                    jaccard_sim = intersection / union if union > 0 else 0
                    jaccard_similarities.append(jaccard_sim)
            
            avg_jaccard_similarity = np.mean(jaccard_similarities) if jaccard_similarities else 0
            diversity_score = 1 - avg_jaccard_similarity
            
            # Count unique features changed
            all_changed_features = set()
            for changes in all_changes:
                all_changed_features.update(changes)
            
            return {
                'diversity_score': float(diversity_score),
                'avg_jaccard_similarity': float(avg_jaccard_similarity),
                'unique_features_changed': len(all_changed_features),
                'total_counterfactuals': len(counterfactuals)
            }
            
        except Exception as e:
            logger.error(f"Error calculating diversity metrics: {e}")
            return {'error': str(e)}
    
    def save_analyzer(self, filepath: str):
        """Save analyzer to file"""
        
        analyzer_data = {
            'feature_names': self.feature_names,
            'feature_ranges': self.feature_ranges,
            'categorical_features': self.categorical_features,
            'scaler': self.scaler,
            'config': self.config
        }
        
        joblib.dump(analyzer_data, filepath)
        logger.info(f"Counterfactual analyzer saved to {filepath}")
    
    def load_analyzer(self, filepath: str):
        """Load analyzer from file"""
        
        analyzer_data = joblib.load(filepath)
        
        self.feature_names = analyzer_data['feature_names']
        self.feature_ranges = analyzer_data['feature_ranges']
        self.categorical_features = analyzer_data['categorical_features']
        self.scaler = analyzer_data['scaler']
        self.config = analyzer_data.get('config', {})
        
        logger.info(f"Counterfactual analyzer loaded from {filepath}")
