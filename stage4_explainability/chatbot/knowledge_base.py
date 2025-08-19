"""
Knowledge base for Stage 4 explainability chatbot.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Knowledge base for credit explanation chatbot"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credit_knowledge = {}
        self.explanation_templates = {}
        self.factor_definitions = {}
        self.improvement_strategies = {}
        self.faq_database = {}
        self.regulatory_info = {}
        self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self):
        """Initialize knowledge base with credit domain knowledge"""
        
        # Credit score factors and definitions
        self.factor_definitions = {
            'payment_history': {
                'name': 'Payment History',
                'weight': 35,
                'description': 'Your track record of making payments on time',
                'impact': 'Most important factor affecting credit score',
                'improvement_tips': [
                    'Always pay at least the minimum amount due',
                    'Set up automatic payments to avoid missed payments',
                    'Pay off past-due accounts as soon as possible',
                    'Contact creditors if you anticipate payment difficulties'
                ]
            },
            'credit_utilization': {
                'name': 'Credit Utilization',
                'weight': 30,
                'description': 'The percentage of available credit you are using',
                'impact': 'Second most important factor',
                'improvement_tips': [
                    'Keep utilization below 30% on all cards',
                    'Aim for under 10% for excellent scores',
                    'Pay down balances before statement dates',
                    'Consider requesting credit limit increases'
                ]
            },
            'length_of_credit_history': {
                'name': 'Length of Credit History',
                'weight': 15,
                'description': 'How long you have been using credit',
                'impact': 'Longer history generally improves scores',
                'improvement_tips': [
                    'Keep old accounts open even if unused',
                    'Use old cards occasionally to keep them active',
                    'Avoid closing your oldest credit accounts',
                    'Be patient - this factor improves with time'
                ]
            },
            'credit_mix': {
                'name': 'Credit Mix',
                'weight': 10,
                'description': 'The variety of credit types you have',
                'impact': 'Having diverse credit types can help',
                'improvement_tips': [
                    'Consider different types of credit (cards, loans, mortgage)',
                    'Only take on credit you need and can manage',
                    'Maintain good payment history across all account types',
                    'Avoid opening accounts just for credit mix'
                ]
            },
            'new_credit': {
                'name': 'New Credit',
                'weight': 10,
                'description': 'Recent credit inquiries and newly opened accounts',
                'impact': 'Too many new accounts can lower scores',
                'improvement_tips': [
                    'Limit credit applications to when necessary',
                    'Space out credit applications over time',
                    'Shop for rates within 14-45 day windows',
                    'Avoid opening multiple accounts quickly'
                ]
            }
        }
        
        # Credit score ranges
        self.credit_knowledge['score_ranges'] = {
            'excellent': {'range': (800, 850), 'description': 'Excellent credit - best rates and terms available'},
            'very_good': {'range': (740, 799), 'description': 'Very good credit - access to favorable rates'},
            'good': {'range': (670, 739), 'description': 'Good credit - near or slightly above average'},
            'fair': {'range': (580, 669), 'description': 'Fair credit - below average, may face higher rates'},
            'poor': {'range': (300, 579), 'description': 'Poor credit - significant challenges getting approved'}
        }
        
        # Loan decision factors
        self.credit_knowledge['loan_factors'] = {
            'debt_to_income_ratio': {
                'name': 'Debt-to-Income Ratio',
                'description': 'Monthly debt payments divided by monthly income',
                'good_range': 'Below 36%',
                'impact': 'Higher ratios indicate higher risk'
            },
            'employment_history': {
                'name': 'Employment History',
                'description': 'Stability and length of employment',
                'good_range': '2+ years with current employer',
                'impact': 'Stable employment reduces lending risk'
            },
            'down_payment': {
                'name': 'Down Payment',
                'description': 'Amount paid upfront for loans like mortgages',
                'good_range': '20% or more for mortgages',
                'impact': 'Larger down payments reduce lender risk'
            },
            'collateral': {
                'name': 'Collateral',
                'description': 'Assets that secure the loan',
                'impact': 'Secured loans are less risky for lenders'
            }
        }
        
        # Improvement strategies
        self.improvement_strategies = {
            'poor_credit': {
                'priority_actions': [
                    'Focus on making all payments on time',
                    'Pay down high-utilization credit cards',
                    'Consider secured credit cards if needed',
                    'Check credit reports for errors and dispute them'
                ],
                'timeline': '6-12 months for initial improvement',
                'expected_improvement': '50-100 points possible'
            },
            'fair_credit': {
                'priority_actions': [
                    'Continue perfect payment history',
                    'Reduce credit utilization below 30%',
                    'Avoid new credit applications',
                    'Pay down existing debt'
                ],
                'timeline': '3-6 months for noticeable improvement',
                'expected_improvement': '30-70 points possible'
            },
            'good_credit': {
                'priority_actions': [
                    'Optimize credit utilization below 10%',
                    'Maintain diverse credit mix',
                    'Keep old accounts open',
                    'Monitor credit reports regularly'
                ],
                'timeline': '2-4 months for optimization',
                'expected_improvement': '20-50 points possible'
            },
            'very_good_credit': {
                'priority_actions': [
                    'Maintain excellent payment history',
                    'Keep utilization very low',
                    'Avoid unnecessary credit applications',
                    'Consider becoming authorized user on old accounts'
                ],
                'timeline': '1-3 months for fine-tuning',
                'expected_improvement': '10-30 points possible'
            }
        }
        
        # FAQ database
        self.faq_database = {
            'what_is_credit_score': {
                'question': 'What is a credit score?',
                'answer': 'A credit score is a three-digit number that represents your creditworthiness. It ranges from 300 to 850, with higher scores indicating better credit. Lenders use this score to evaluate the risk of lending to you.',
                'category': 'basics'
            },
            'how_is_credit_score_calculated': {
                'question': 'How is my credit score calculated?',
                'answer': 'Credit scores are calculated using five main factors: Payment History (35%), Credit Utilization (30%), Length of Credit History (15%), Credit Mix (10%), and New Credit (10%). The exact algorithms vary by scoring model.',
                'category': 'calculation'
            },
            'how_often_does_score_change': {
                'question': 'How often does my credit score change?',
                'answer': 'Credit scores can change whenever your credit report is updated, which typically happens monthly when creditors report new information. Significant changes may take 1-2 months to appear.',
                'category': 'updates'
            },
            'what_hurts_credit_score': {
                'question': 'What hurts my credit score the most?',
                'answer': 'Late payments and high credit utilization have the biggest negative impact. Other factors include collections, bankruptcies, foreclosures, and too many hard inquiries in a short period.',
                'category': 'negative_factors'
            },
            'how_to_improve_quickly': {
                'question': 'How can I improve my credit score quickly?',
                'answer': 'The fastest improvements come from paying down credit card balances to reduce utilization and ensuring all payments are made on time. Some changes can be seen within 30-60 days.',
                'category': 'improvement'
            }
        }
        
        # Regulatory and compliance information
        self.regulatory_info = {
            'fair_credit_reporting_act': {
                'name': 'Fair Credit Reporting Act (FCRA)',
                'description': 'Federal law governing credit reporting and consumer rights',
                'key_rights': [
                    'Right to free annual credit reports',
                    'Right to dispute inaccurate information',
                    'Right to know who has accessed your credit',
                    'Right to opt out of prescreened offers'
                ]
            },
            'equal_credit_opportunity_act': {
                'name': 'Equal Credit Opportunity Act (ECOA)',
                'description': 'Prohibits discrimination in credit decisions',
                'protected_classes': [
                    'Race', 'Color', 'Religion', 'National origin',
                    'Sex', 'Marital status', 'Age', 'Public assistance receipt'
                ]
            }
        }
    
    async def get_factor_explanation(self, factor_name: str) -> Dict[str, Any]:
        """Get detailed explanation of a credit factor"""
        
        try:
            factor_key = factor_name.lower().replace(' ', '_')
            
            if factor_key in self.factor_definitions:
                factor_info = self.factor_definitions[factor_key].copy()
                factor_info['timestamp'] = datetime.now().isoformat()
                return factor_info
            
            # Try partial matching
            for key, info in self.factor_definitions.items():
                if factor_name.lower() in key or factor_name.lower() in info['name'].lower():
                    factor_info = info.copy()
                    factor_info['timestamp'] = datetime.now().isoformat()
                    return factor_info
            
            return {'error': f'Factor {factor_name} not found in knowledge base'}
            
        except Exception as e:
            logger.error(f"Error getting factor explanation: {e}")
            return {'error': str(e)}
    
    async def get_improvement_strategy(self, current_score: int, target_score: int = None) -> Dict[str, Any]:
        """Get improvement strategy based on current credit score"""
        
        try:
            # Determine credit category
            if current_score >= 740:
                category = 'very_good_credit'
            elif current_score >= 670:
                category = 'good_credit'
            elif current_score >= 580:
                category = 'fair_credit'
            else:
                category = 'poor_credit'
            
            strategy = self.improvement_strategies[category].copy()
            
            # Customize based on target score
            if target_score:
                score_gap = target_score - current_score
                if score_gap > 100:
                    strategy['timeline'] = '12-24 months for significant improvement'
                    strategy['additional_notes'] = 'Large score improvements require patience and consistent effort'
                elif score_gap > 50:
                    strategy['timeline'] = '6-12 months for substantial improvement'
                else:
                    strategy['timeline'] = '3-6 months for moderate improvement'
            
            strategy['current_category'] = category
            strategy['current_score'] = current_score
            strategy['target_score'] = target_score
            strategy['timestamp'] = datetime.now().isoformat()
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error getting improvement strategy: {e}")
            return {'error': str(e)}
    
    async def search_faq(self, query: str) -> List[Dict[str, Any]]:
        """Search FAQ database for relevant answers"""
        
        try:
            query_lower = query.lower()
            results = []
            
            for faq_id, faq_data in self.faq_database.items():
                # Check if query matches question or answer
                question_match = any(word in faq_data['question'].lower() for word in query_lower.split())
                answer_match = any(word in faq_data['answer'].lower() for word in query_lower.split())
                
                if question_match or answer_match:
                    # Calculate relevance score
                    score = 0
                    for word in query_lower.split():
                        if word in faq_data['question'].lower():
                            score += 2
                        if word in faq_data['answer'].lower():
                            score += 1
                    
                    result = faq_data.copy()
                    result['relevance_score'] = score
                    result['faq_id'] = faq_id
                    results.append(result)
            
            # Sort by relevance
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return results[:5]  # Return top 5 results
            
        except Exception as e:
            logger.error(f"Error searching FAQ: {e}")
            return []
    
    async def get_score_range_info(self, score: int) -> Dict[str, Any]:
        """Get information about credit score range"""
        
        try:
            for range_name, range_info in self.credit_knowledge['score_ranges'].items():
                min_score, max_score = range_info['range']
                if min_score <= score <= max_score:
                    result = range_info.copy()
                    result['range_name'] = range_name
                    result['score'] = score
                    result['timestamp'] = datetime.now().isoformat()
                    return result
            
            return {'error': f'Score {score} is outside normal range (300-850)'}
            
        except Exception as e:
            logger.error(f"Error getting score range info: {e}")
            return {'error': str(e)}
    
    async def get_loan_factor_explanation(self, factor_name: str) -> Dict[str, Any]:
        """Get explanation of loan decision factors"""
        
        try:
            factor_key = factor_name.lower().replace(' ', '_').replace('-', '_')
            
            if factor_key in self.credit_knowledge['loan_factors']:
                factor_info = self.credit_knowledge['loan_factors'][factor_key].copy()
                factor_info['timestamp'] = datetime.now().isoformat()
                return factor_info
            
            # Try partial matching
            for key, info in self.credit_knowledge['loan_factors'].items():
                if factor_name.lower() in key or factor_name.lower() in info['name'].lower():
                    factor_info = info.copy()
                    factor_info['timestamp'] = datetime.now().isoformat()
                    return factor_info
            
            return {'error': f'Loan factor {factor_name} not found in knowledge base'}
            
        except Exception as e:
            logger.error(f"Error getting loan factor explanation: {e}")
            return {'error': str(e)}
    
    async def get_regulatory_info(self, topic: str) -> Dict[str, Any]:
        """Get regulatory and compliance information"""
        
        try:
            topic_key = topic.lower().replace(' ', '_')
            
            for key, info in self.regulatory_info.items():
                if topic_key in key or topic.lower() in info['name'].lower():
                    result = info.copy()
                    result['timestamp'] = datetime.now().isoformat()
                    return result
            
            return {'error': f'Regulatory topic {topic} not found'}
            
        except Exception as e:
            logger.error(f"Error getting regulatory info: {e}")
            return {'error': str(e)}
    
    async def get_personalized_explanation(self, topic: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get personalized explanation based on user context"""
        
        try:
            base_explanation = await self.get_factor_explanation(topic)
            
            if 'error' in base_explanation:
                return base_explanation
            
            # Personalize based on user context
            user_score = user_context.get('credit_score')
            user_utilization = user_context.get('credit_utilization')
            technical_comfort = user_context.get('technical_comfort', 'medium')
            
            personalized = base_explanation.copy()
            
            # Add personalized insights
            if topic.lower() == 'credit_utilization' and user_utilization:
                if user_utilization > 30:
                    personalized['personal_insight'] = f"Your current utilization of {user_utilization}% is above the recommended 30%. This is likely impacting your score negatively."
                elif user_utilization < 10:
                    personalized['personal_insight'] = f"Your utilization of {user_utilization}% is excellent and helping your score."
                else:
                    personalized['personal_insight'] = f"Your utilization of {user_utilization}% is in a good range but could be optimized further."
            
            # Adjust explanation complexity
            if technical_comfort == 'low':
                personalized['simple_explanation'] = self._simplify_explanation(base_explanation['description'])
            elif technical_comfort == 'high':
                personalized['technical_details'] = self._add_technical_details(topic)
            
            return personalized
            
        except Exception as e:
            logger.error(f"Error getting personalized explanation: {e}")
            return {'error': str(e)}
    
    def _simplify_explanation(self, explanation: str) -> str:
        """Simplify explanation for users with low technical comfort"""
        
        # Simple mapping of complex terms
        simplifications = {
            'creditworthiness': 'how likely you are to pay back money you borrow',
            'utilization': 'how much of your available credit you are using',
            'algorithm': 'calculation method',
            'factor': 'thing that affects your score'
        }
        
        simplified = explanation
        for complex_term, simple_term in simplifications.items():
            simplified = simplified.replace(complex_term, simple_term)
        
        return simplified
    
    def _add_technical_details(self, topic: str) -> str:
        """Add technical details for users with high technical comfort"""
        
        technical_details = {
            'payment_history': 'Uses weighted scoring based on recency, severity, and frequency of late payments. Recent late payments have higher impact than older ones.',
            'credit_utilization': 'Calculated both per-card and overall. Both individual card utilization and aggregate utilization across all cards are considered.',
            'credit_mix': 'Evaluates the diversity of credit types including revolving credit (credit cards) and installment loans (mortgages, auto loans).'
        }
        
        return technical_details.get(topic.lower().replace(' ', '_'), 'Additional technical details not available for this topic.')
    
    async def add_knowledge_item(self, category: str, key: str, data: Dict[str, Any]):
        """Add new knowledge item to the knowledge base"""
        
        try:
            if category == 'factors':
                self.factor_definitions[key] = data
            elif category == 'faq':
                self.faq_database[key] = data
            elif category == 'strategies':
                self.improvement_strategies[key] = data
            elif category == 'regulatory':
                self.regulatory_info[key] = data
            else:
                if category not in self.credit_knowledge:
                    self.credit_knowledge[category] = {}
                self.credit_knowledge[category][key] = data
            
            logger.info(f"Added knowledge item: {category}/{key}")
            
        except Exception as e:
            logger.error(f"Error adding knowledge item: {e}")
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        
        return {
            'factor_definitions': len(self.factor_definitions),
            'faq_items': len(self.faq_database),
            'improvement_strategies': len(self.improvement_strategies),
            'regulatory_items': len(self.regulatory_info),
            'score_ranges': len(self.credit_knowledge.get('score_ranges', {})),
            'loan_factors': len(self.credit_knowledge.get('loan_factors', {})),
            'total_knowledge_items': (
                len(self.factor_definitions) + 
                len(self.faq_database) + 
                len(self.improvement_strategies) + 
                len(self.regulatory_info)
            ),
            'timestamp': datetime.now().isoformat()
        }
