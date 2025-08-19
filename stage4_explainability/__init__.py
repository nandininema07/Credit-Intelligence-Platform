"""
Stage 4: Explainability & Chatbot Integration
Model explainability, SHAP analysis, and chatbot integration for credit decisions.
"""

from .explainer.shap_explainer import SHAPExplainer
from .chatbot.credit_chatbot import CreditChatbot

__all__ = ['SHAPExplainer', 'CreditChatbot']
