"""
Visualization data generator for Stage 4 explainability.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
from enum import Enum

logger = logging.getLogger(__name__)

class VisualizationType(Enum):
    """Types of visualizations"""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    WATERFALL = "waterfall"
    GAUGE = "gauge"
    RADAR = "radar"
    TREEMAP = "treemap"

class VisualizationDataGenerator:
    """Generate visualization data for explanations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.color_schemes = {
            'importance': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'impact': ['#ff4444', '#ffaa44', '#44ff44', '#4444ff', '#aa44ff'],
            'neutral': ['#666666', '#888888', '#aaaaaa', '#cccccc', '#eeeeee']
        }
        
    async def generate_visualization_data(self, explanation_type, explanation_data: Dict[str, Any],
                                        preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate visualization data based on explanation type"""
        
        try:
            preferences = preferences or {}
            
            if hasattr(explanation_type, 'value'):
                explanation_type = explanation_type.value
            
            if explanation_type == 'local':
                return await self._generate_local_visualizations(explanation_data, preferences)
            elif explanation_type == 'global':
                return await self._generate_global_visualizations(explanation_data, preferences)
            elif explanation_type == 'counterfactual':
                return await self._generate_counterfactual_visualizations(explanation_data, preferences)
            elif explanation_type == 'feature_importance':
                return await self._generate_importance_visualizations(explanation_data, preferences)
            elif explanation_type == 'what_if':
                return await self._generate_whatif_visualizations(explanation_data, preferences)
            else:
                return await self._generate_default_visualizations(explanation_data, preferences)
                
        except Exception as e:
            logger.error(f"Error generating visualization data: {e}")
            return {'error': str(e)}
    
    async def _generate_local_visualizations(self, explanation_data: Dict[str, Any],
                                           preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualizations for local explanations"""
        
        try:
            visualizations = {}
            
            # Feature importance bar chart
            if 'combined_importance' in explanation_data:
                importance = explanation_data['combined_importance']
                visualizations['feature_importance_bar'] = self._create_bar_chart_data(
                    importance, 'Feature Importance', 'Features', 'Importance Score'
                )
                
                # Waterfall chart for feature contributions
                visualizations['contribution_waterfall'] = self._create_waterfall_data(
                    importance, 'Feature Contributions'
                )
            
            # Gauge chart for prediction confidence
            prediction_confidence = explanation_data.get('prediction_confidence', 0.8)
            visualizations['confidence_gauge'] = self._create_gauge_data(
                prediction_confidence, 'Prediction Confidence', 0, 1
            )
            
            return {
                'visualizations': visualizations,
                'layout_suggestions': self._suggest_layout('local'),
                'interactive_features': ['hover_details', 'zoom', 'filter'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating local visualizations: {e}")
            return {'error': str(e)}
    
    async def _generate_global_visualizations(self, explanation_data: Dict[str, Any],
                                            preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualizations for global explanations"""
        
        try:
            visualizations = {}
            
            # Global feature importance
            if 'explanations' in explanation_data and 'global_importance' in explanation_data['explanations']:
                global_data = explanation_data['explanations']['global_importance']
                
                if 'importance_methods' in global_data:
                    consensus = global_data['importance_methods'].get('consensus', {})
                    if consensus:
                        visualizations['global_importance_bar'] = self._create_bar_chart_data(
                            consensus, 'Global Feature Importance', 'Features', 'Importance'
                        )
            
            return {
                'visualizations': visualizations,
                'layout_suggestions': self._suggest_layout('global'),
                'interactive_features': ['hover_details', 'zoom', 'filter', 'drill_down'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating global visualizations: {e}")
            return {'error': str(e)}
    
    async def _generate_counterfactual_visualizations(self, explanation_data: Dict[str, Any],
                                                    preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualizations for counterfactual explanations"""
        
        try:
            visualizations = {}
            
            # Before/after comparison
            if 'counterfactuals' in explanation_data and explanation_data['counterfactuals']:
                cf_data = explanation_data['counterfactuals'][0]
                
                if 'changes_made' in cf_data:
                    changes = cf_data['changes_made']
                    change_data = {feature: info['relative_change'] 
                                 for feature, info in changes.items()}
                    
                    visualizations['changes_bar'] = self._create_bar_chart_data(
                        change_data, 'Required Changes', 'Features', 'Relative Change (%)'
                    )
            
            return {
                'visualizations': visualizations,
                'layout_suggestions': self._suggest_layout('counterfactual'),
                'interactive_features': ['hover_details', 'compare', 'animate'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating counterfactual visualizations: {e}")
            return {'error': str(e)}
    
    async def _generate_importance_visualizations(self, explanation_data: Dict[str, Any],
                                                preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualizations for feature importance"""
        
        try:
            visualizations = {}
            
            # Consensus importance ranking
            if 'consensus_importance' in explanation_data:
                consensus = explanation_data['consensus_importance']
                
                if 'consensus_scores' in consensus:
                    scores = consensus['consensus_scores']
                    visualizations['importance_ranking_bar'] = self._create_horizontal_bar_data(
                        scores, 'Feature Importance Ranking', 'Importance Score', 'Features'
                    )
            
            return {
                'visualizations': visualizations,
                'layout_suggestions': self._suggest_layout('importance'),
                'interactive_features': ['hover_details', 'sort', 'filter'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating importance visualizations: {e}")
            return {'error': str(e)}
    
    async def _generate_whatif_visualizations(self, explanation_data: Dict[str, Any],
                                            preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualizations for what-if scenarios"""
        
        try:
            visualizations = {}
            
            # Scenario impact comparison
            if 'ranked_scenarios' in explanation_data:
                scenarios = explanation_data['ranked_scenarios']
                
                scenario_impacts = {}
                for scenario in scenarios:
                    name = scenario.get('scenario', {}).get('name', 'Scenario')
                    impact = scenario.get('impact_score', 0)
                    scenario_impacts[name] = impact
                
                visualizations['scenario_impact_bar'] = self._create_bar_chart_data(
                    scenario_impacts, 'Scenario Impact Comparison', 'Scenarios', 'Impact Score'
                )
            
            return {
                'visualizations': visualizations,
                'layout_suggestions': self._suggest_layout('what_if'),
                'interactive_features': ['hover_details', 'compare', 'filter', 'animate'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating what-if visualizations: {e}")
            return {'error': str(e)}
    
    async def _generate_default_visualizations(self, explanation_data: Dict[str, Any],
                                             preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate default visualizations for unknown explanation types"""
        
        return {
            'visualizations': {
                'summary_text': {
                    'type': 'text',
                    'content': 'Visualization data available for supported explanation types'
                }
            },
            'layout_suggestions': {'layout': 'single_column'},
            'interactive_features': [],
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_bar_chart_data(self, data: Dict[str, float], title: str,
                              x_label: str, y_label: str) -> Dict[str, Any]:
        """Create bar chart data structure"""
        
        sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'type': VisualizationType.BAR_CHART.value,
            'title': title,
            'data': {
                'labels': [item[0] for item in sorted_data],
                'values': [item[1] for item in sorted_data],
                'colors': self.color_schemes['importance'][:len(sorted_data)]
            },
            'layout': {
                'x_axis': {'title': x_label},
                'y_axis': {'title': y_label},
                'show_values': True
            }
        }
    
    def _create_horizontal_bar_data(self, data: Dict[str, float], title: str,
                                   x_label: str, y_label: str) -> Dict[str, Any]:
        """Create horizontal bar chart data"""
        
        sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=False)
        
        return {
            'type': 'horizontal_bar',
            'title': title,
            'data': {
                'labels': [item[0] for item in sorted_data],
                'values': [item[1] for item in sorted_data],
                'colors': self.color_schemes['importance'][:len(sorted_data)]
            },
            'layout': {
                'x_axis': {'title': x_label},
                'y_axis': {'title': y_label}
            }
        }
    
    def _create_waterfall_data(self, importance: Dict[str, float], title: str) -> Dict[str, Any]:
        """Create waterfall chart data"""
        
        sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'type': VisualizationType.WATERFALL.value,
            'title': title,
            'data': {
                'categories': [item[0] for item in sorted_importance],
                'values': [item[1] for item in sorted_importance],
                'colors': [self.color_schemes['impact'][0] if val > 0 else self.color_schemes['impact'][1] 
                          for _, val in sorted_importance]
            },
            'layout': {
                'show_connectors': True,
                'show_total': True
            }
        }
    
    def _create_gauge_data(self, value: float, title: str, min_val: float, max_val: float) -> Dict[str, Any]:
        """Create gauge chart data"""
        
        return {
            'type': VisualizationType.GAUGE.value,
            'title': title,
            'data': {
                'value': value,
                'min': min_val,
                'max': max_val,
                'ranges': [
                    {'from': min_val, 'to': (max_val - min_val) * 0.33 + min_val, 'color': '#ff4444'},
                    {'from': (max_val - min_val) * 0.33 + min_val, 'to': (max_val - min_val) * 0.66 + min_val, 'color': '#ffaa44'},
                    {'from': (max_val - min_val) * 0.66 + min_val, 'to': max_val, 'color': '#44ff44'}
                ]
            },
            'layout': {
                'show_value': True,
                'show_ranges': True
            }
        }
    
    def _suggest_layout(self, explanation_type: str) -> Dict[str, Any]:
        """Suggest layout for visualization type"""
        
        layouts = {
            'local': {'layout': 'two_column', 'primary': 'feature_importance_bar'},
            'global': {'layout': 'grid', 'columns': 2},
            'counterfactual': {'layout': 'comparison', 'split': 'horizontal'},
            'importance': {'layout': 'single_column', 'primary': 'importance_ranking_bar'},
            'what_if': {'layout': 'dashboard', 'sections': 3}
        }
        
        return layouts.get(explanation_type, {'layout': 'single_column'})
    
    def get_visualization_statistics(self) -> Dict[str, Any]:
        """Get visualization generator statistics"""
        
        return {
            'supported_types': [t.value for t in VisualizationType],
            'color_schemes': list(self.color_schemes.keys()),
            'interactive_features': ['hover_details', 'zoom', 'filter', 'compare', 'animate', 'drill_down'],
            'timestamp': datetime.now().isoformat()
        }
