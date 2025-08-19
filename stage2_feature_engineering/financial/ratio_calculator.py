"""
Financial ratio calculator for comprehensive financial analysis.
Calculates liquidity, profitability, leverage, and efficiency ratios.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FinancialRatios:
    """Financial ratios data structure"""
    liquidity_ratios: Dict[str, float]
    profitability_ratios: Dict[str, float]
    leverage_ratios: Dict[str, float]
    efficiency_ratios: Dict[str, float]
    market_ratios: Dict[str, float]
    calculation_date: datetime

class RatioCalculator:
    """Financial ratio calculator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def calculate_ratios(self, financial_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate all financial ratios from financial data"""
        if not financial_data:
            return {}
        
        # Aggregate financial data
        aggregated_data = self._aggregate_financial_data(financial_data)
        
        # Calculate different ratio categories
        liquidity_ratios = self._calculate_liquidity_ratios(aggregated_data)
        profitability_ratios = self._calculate_profitability_ratios(aggregated_data)
        leverage_ratios = self._calculate_leverage_ratios(aggregated_data)
        efficiency_ratios = self._calculate_efficiency_ratios(aggregated_data)
        market_ratios = self._calculate_market_ratios(aggregated_data)
        
        # Combine all ratios
        all_ratios = {}
        all_ratios.update({f'liquidity_{k}': v for k, v in liquidity_ratios.items()})
        all_ratios.update({f'profitability_{k}': v for k, v in profitability_ratios.items()})
        all_ratios.update({f'leverage_{k}': v for k, v in leverage_ratios.items()})
        all_ratios.update({f'efficiency_{k}': v for k, v in efficiency_ratios.items()})
        all_ratios.update({f'market_{k}': v for k, v in market_ratios.items()})
        
        logger.info(f"Calculated {len(all_ratios)} financial ratios")
        return all_ratios
    
    def _aggregate_financial_data(self, financial_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate financial data from multiple sources"""
        aggregated = {
            'current_assets': 0.0,
            'current_liabilities': 0.0,
            'total_assets': 0.0,
            'total_liabilities': 0.0,
            'total_equity': 0.0,
            'revenue': 0.0,
            'net_income': 0.0,
            'gross_profit': 0.0,
            'operating_income': 0.0,
            'ebitda': 0.0,
            'cash': 0.0,
            'inventory': 0.0,
            'accounts_receivable': 0.0,
            'accounts_payable': 0.0,
            'long_term_debt': 0.0,
            'short_term_debt': 0.0,
            'market_cap': 0.0,
            'stock_price': 0.0,
            'shares_outstanding': 0.0,
            'cost_of_goods_sold': 0.0
        }
        
        # Extract and aggregate values from financial data
        for data_point in financial_data:
            for key in aggregated.keys():
                value = data_point.get(key, 0)
                if isinstance(value, (int, float)) and not np.isnan(value):
                    aggregated[key] += value
        
        # Calculate derived values
        if aggregated['total_assets'] == 0 and aggregated['total_liabilities'] > 0:
            aggregated['total_assets'] = aggregated['total_liabilities'] + aggregated['total_equity']
        
        if aggregated['gross_profit'] == 0 and aggregated['revenue'] > 0 and aggregated['cost_of_goods_sold'] > 0:
            aggregated['gross_profit'] = aggregated['revenue'] - aggregated['cost_of_goods_sold']
        
        return aggregated
    
    def _calculate_liquidity_ratios(self, data: Dict[str, float]) -> Dict[str, float]:
        """Calculate liquidity ratios"""
        ratios = {}
        
        # Current Ratio
        if data['current_liabilities'] > 0:
            ratios['current_ratio'] = data['current_assets'] / data['current_liabilities']
        else:
            ratios['current_ratio'] = 0.0
        
        # Quick Ratio (Acid Test)
        quick_assets = data['current_assets'] - data['inventory']
        if data['current_liabilities'] > 0:
            ratios['quick_ratio'] = quick_assets / data['current_liabilities']
        else:
            ratios['quick_ratio'] = 0.0
        
        # Cash Ratio
        if data['current_liabilities'] > 0:
            ratios['cash_ratio'] = data['cash'] / data['current_liabilities']
        else:
            ratios['cash_ratio'] = 0.0
        
        # Working Capital
        ratios['working_capital'] = data['current_assets'] - data['current_liabilities']
        
        # Working Capital Ratio
        if data['total_assets'] > 0:
            ratios['working_capital_ratio'] = ratios['working_capital'] / data['total_assets']
        else:
            ratios['working_capital_ratio'] = 0.0
        
        return ratios
    
    def _calculate_profitability_ratios(self, data: Dict[str, float]) -> Dict[str, float]:
        """Calculate profitability ratios"""
        ratios = {}
        
        # Gross Profit Margin
        if data['revenue'] > 0:
            ratios['gross_profit_margin'] = data['gross_profit'] / data['revenue']
        else:
            ratios['gross_profit_margin'] = 0.0
        
        # Operating Profit Margin
        if data['revenue'] > 0:
            ratios['operating_profit_margin'] = data['operating_income'] / data['revenue']
        else:
            ratios['operating_profit_margin'] = 0.0
        
        # Net Profit Margin
        if data['revenue'] > 0:
            ratios['net_profit_margin'] = data['net_income'] / data['revenue']
        else:
            ratios['net_profit_margin'] = 0.0
        
        # Return on Assets (ROA)
        if data['total_assets'] > 0:
            ratios['return_on_assets'] = data['net_income'] / data['total_assets']
        else:
            ratios['return_on_assets'] = 0.0
        
        # Return on Equity (ROE)
        if data['total_equity'] > 0:
            ratios['return_on_equity'] = data['net_income'] / data['total_equity']
        else:
            ratios['return_on_equity'] = 0.0
        
        # EBITDA Margin
        if data['revenue'] > 0:
            ratios['ebitda_margin'] = data['ebitda'] / data['revenue']
        else:
            ratios['ebitda_margin'] = 0.0
        
        return ratios
    
    def _calculate_leverage_ratios(self, data: Dict[str, float]) -> Dict[str, float]:
        """Calculate leverage/debt ratios"""
        ratios = {}
        
        # Debt-to-Equity Ratio
        total_debt = data['long_term_debt'] + data['short_term_debt']
        if data['total_equity'] > 0:
            ratios['debt_to_equity'] = total_debt / data['total_equity']
        else:
            ratios['debt_to_equity'] = 0.0
        
        # Debt-to-Assets Ratio
        if data['total_assets'] > 0:
            ratios['debt_to_assets'] = total_debt / data['total_assets']
        else:
            ratios['debt_to_assets'] = 0.0
        
        # Equity Multiplier
        if data['total_equity'] > 0:
            ratios['equity_multiplier'] = data['total_assets'] / data['total_equity']
        else:
            ratios['equity_multiplier'] = 0.0
        
        # Interest Coverage Ratio
        # Note: Would need interest expense data
        ratios['interest_coverage'] = 0.0  # Placeholder
        
        # Debt Service Coverage Ratio
        # Note: Would need debt service payments data
        ratios['debt_service_coverage'] = 0.0  # Placeholder
        
        return ratios
    
    def _calculate_efficiency_ratios(self, data: Dict[str, float]) -> Dict[str, float]:
        """Calculate efficiency/activity ratios"""
        ratios = {}
        
        # Asset Turnover
        if data['total_assets'] > 0:
            ratios['asset_turnover'] = data['revenue'] / data['total_assets']
        else:
            ratios['asset_turnover'] = 0.0
        
        # Inventory Turnover
        if data['inventory'] > 0:
            ratios['inventory_turnover'] = data['cost_of_goods_sold'] / data['inventory']
        else:
            ratios['inventory_turnover'] = 0.0
        
        # Receivables Turnover
        if data['accounts_receivable'] > 0:
            ratios['receivables_turnover'] = data['revenue'] / data['accounts_receivable']
        else:
            ratios['receivables_turnover'] = 0.0
        
        # Payables Turnover
        if data['accounts_payable'] > 0:
            ratios['payables_turnover'] = data['cost_of_goods_sold'] / data['accounts_payable']
        else:
            ratios['payables_turnover'] = 0.0
        
        # Days Sales Outstanding (DSO)
        if ratios['receivables_turnover'] > 0:
            ratios['days_sales_outstanding'] = 365 / ratios['receivables_turnover']
        else:
            ratios['days_sales_outstanding'] = 0.0
        
        # Days Inventory Outstanding (DIO)
        if ratios['inventory_turnover'] > 0:
            ratios['days_inventory_outstanding'] = 365 / ratios['inventory_turnover']
        else:
            ratios['days_inventory_outstanding'] = 0.0
        
        # Days Payable Outstanding (DPO)
        if ratios['payables_turnover'] > 0:
            ratios['days_payable_outstanding'] = 365 / ratios['payables_turnover']
        else:
            ratios['days_payable_outstanding'] = 0.0
        
        # Cash Conversion Cycle
        ratios['cash_conversion_cycle'] = (ratios['days_sales_outstanding'] + 
                                         ratios['days_inventory_outstanding'] - 
                                         ratios['days_payable_outstanding'])
        
        return ratios
    
    def _calculate_market_ratios(self, data: Dict[str, float]) -> Dict[str, float]:
        """Calculate market-based ratios"""
        ratios = {}
        
        # Earnings Per Share (EPS)
        if data['shares_outstanding'] > 0:
            ratios['earnings_per_share'] = data['net_income'] / data['shares_outstanding']
        else:
            ratios['earnings_per_share'] = 0.0
        
        # Price-to-Earnings Ratio (P/E)
        if ratios['earnings_per_share'] > 0:
            ratios['price_to_earnings'] = data['stock_price'] / ratios['earnings_per_share']
        else:
            ratios['price_to_earnings'] = 0.0
        
        # Price-to-Book Ratio (P/B)
        book_value_per_share = data['total_equity'] / data['shares_outstanding'] if data['shares_outstanding'] > 0 else 0
        if book_value_per_share > 0:
            ratios['price_to_book'] = data['stock_price'] / book_value_per_share
        else:
            ratios['price_to_book'] = 0.0
        
        # Market-to-Book Ratio
        ratios['market_to_book'] = ratios['price_to_book']
        
        # Price-to-Sales Ratio
        sales_per_share = data['revenue'] / data['shares_outstanding'] if data['shares_outstanding'] > 0 else 0
        if sales_per_share > 0:
            ratios['price_to_sales'] = data['stock_price'] / sales_per_share
        else:
            ratios['price_to_sales'] = 0.0
        
        # Enterprise Value to EBITDA
        enterprise_value = data['market_cap'] + (data['long_term_debt'] + data['short_term_debt']) - data['cash']
        if data['ebitda'] > 0:
            ratios['ev_to_ebitda'] = enterprise_value / data['ebitda']
        else:
            ratios['ev_to_ebitda'] = 0.0
        
        return ratios
    
    def calculate_ratio_trends(self, historical_ratios: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate trends in financial ratios over time"""
        if len(historical_ratios) < 2:
            return {}
        
        trends = {}
        
        # Get all ratio names from the latest data
        latest_ratios = historical_ratios[-1]
        
        for ratio_name in latest_ratios.keys():
            values = []
            for ratio_data in historical_ratios:
                if ratio_name in ratio_data:
                    values.append(ratio_data[ratio_name])
            
            if len(values) >= 2:
                # Calculate trend metrics
                trend_data = {
                    'current_value': values[-1],
                    'previous_value': values[-2] if len(values) > 1 else values[-1],
                    'change': values[-1] - values[-2] if len(values) > 1 else 0,
                    'percent_change': ((values[-1] - values[-2]) / values[-2] * 100) if len(values) > 1 and values[-2] != 0 else 0,
                    'trend_direction': 'improving' if values[-1] > values[-2] else 'declining' if values[-1] < values[-2] else 'stable',
                    'volatility': np.std(values) if len(values) > 2 else 0,
                    'average': np.mean(values),
                    'min_value': min(values),
                    'max_value': max(values)
                }
                
                trends[ratio_name] = trend_data
        
        return trends
    
    def benchmark_ratios(self, ratios: Dict[str, float], industry_benchmarks: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compare ratios against industry benchmarks"""
        benchmarked_ratios = {}
        
        for ratio_name, ratio_value in ratios.items():
            if ratio_name in industry_benchmarks:
                benchmark_data = industry_benchmarks[ratio_name]
                
                benchmarked_ratios[ratio_name] = {
                    'value': ratio_value,
                    'industry_median': benchmark_data.get('median', 0),
                    'industry_25th_percentile': benchmark_data.get('p25', 0),
                    'industry_75th_percentile': benchmark_data.get('p75', 0),
                    'percentile_rank': self._calculate_percentile_rank(ratio_value, benchmark_data),
                    'performance': self._assess_performance(ratio_value, benchmark_data, ratio_name)
                }
        
        return benchmarked_ratios
    
    def _calculate_percentile_rank(self, value: float, benchmark_data: Dict[str, float]) -> float:
        """Calculate percentile rank against industry benchmarks"""
        # Simplified percentile calculation
        median = benchmark_data.get('median', 0)
        p25 = benchmark_data.get('p25', 0)
        p75 = benchmark_data.get('p75', 0)
        
        if value <= p25:
            return 25.0
        elif value <= median:
            return 25.0 + (value - p25) / (median - p25) * 25.0
        elif value <= p75:
            return 50.0 + (value - median) / (p75 - median) * 25.0
        else:
            return 75.0
    
    def _assess_performance(self, value: float, benchmark_data: Dict[str, float], ratio_name: str) -> str:
        """Assess performance relative to benchmarks"""
        median = benchmark_data.get('median', 0)
        
        # Define which ratios are "higher is better" vs "lower is better"
        higher_is_better = [
            'current_ratio', 'quick_ratio', 'gross_profit_margin', 'net_profit_margin',
            'return_on_assets', 'return_on_equity', 'asset_turnover'
        ]
        
        lower_is_better = [
            'debt_to_equity', 'debt_to_assets', 'days_sales_outstanding',
            'days_inventory_outstanding', 'cash_conversion_cycle'
        ]
        
        if any(ratio_type in ratio_name for ratio_type in higher_is_better):
            if value > median * 1.1:
                return 'excellent'
            elif value > median:
                return 'good'
            elif value > median * 0.9:
                return 'average'
            else:
                return 'poor'
        elif any(ratio_type in ratio_name for ratio_type in lower_is_better):
            if value < median * 0.9:
                return 'excellent'
            elif value < median:
                return 'good'
            elif value < median * 1.1:
                return 'average'
            else:
                return 'poor'
        else:
            return 'average'
