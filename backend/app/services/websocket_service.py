"""
WebSocket service for real-time communication
"""

from fastapi import WebSocket
from typing import Dict, List, Set, Any, Optional
import json
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_types: Dict[str, str] = {}
        self.subscriptions: Dict[str, Set[int]] = {}  # client_id -> company_ids
        self.session_connections: Dict[str, str] = {}  # session_id -> client_id
    
    async def connect(self, websocket: WebSocket, client_id: str, connection_type: str, session_id: Optional[str] = None):
        """Connect a new WebSocket client"""
        self.active_connections[client_id] = websocket
        self.connection_types[client_id] = connection_type
        self.subscriptions[client_id] = set()
        
        if session_id:
            self.session_connections[session_id] = client_id
        
        logger.info(f"WebSocket client {client_id} connected ({connection_type})")
    
    def disconnect(self, client_id: str):
        """Disconnect a WebSocket client"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_types:
            del self.connection_types[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        
        # Remove from session connections
        session_to_remove = None
        for session_id, cid in self.session_connections.items():
            if cid == client_id:
                session_to_remove = session_id
                break
        if session_to_remove:
            del self.session_connections[session_to_remove]
        
        logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_personal_message(self, message: str, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {str(e)}")
                self.disconnect(client_id)
    
    async def broadcast_to_type(self, message: str, connection_type: str):
        """Broadcast message to all clients of specific type"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            if self.connection_types.get(client_id) == connection_type:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {str(e)}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def broadcast_to_subscribers(self, message: str, company_id: int):
        """Broadcast message to clients subscribed to specific company"""
        disconnected_clients = []
        
        for client_id, company_ids in self.subscriptions.items():
            if company_id in company_ids and client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to subscriber {client_id}: {str(e)}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def subscribe_to_companies(self, client_id: str, company_ids: List[int]):
        """Subscribe client to company updates"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].update(company_ids)
            logger.info(f"Client {client_id} subscribed to companies: {company_ids}")
    
    def unsubscribe_from_companies(self, client_id: str, company_ids: List[int]):
        """Unsubscribe client from company updates"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id] -= set(company_ids)
            logger.info(f"Client {client_id} unsubscribed from companies: {company_ids}")
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return len(self.active_connections)
    
    def get_connections_by_type(self) -> Dict[str, int]:
        """Get connection count by type"""
        type_counts = {}
        for connection_type in self.connection_types.values():
            type_counts[connection_type] = type_counts.get(connection_type, 0) + 1
        return type_counts
    
    def get_active_subscriptions(self) -> Dict[str, int]:
        """Get active subscription counts"""
        company_subscription_counts = {}
        for company_ids in self.subscriptions.values():
            for company_id in company_ids:
                company_subscription_counts[str(company_id)] = company_subscription_counts.get(str(company_id), 0) + 1
        return company_subscription_counts

class WebSocketService:
    """Service for WebSocket operations"""
    
    def __init__(self):
        self.manager = ConnectionManager()
    
    async def connect(self, websocket: WebSocket, client_id: str, connection_type: str, session_id: Optional[str] = None):
        """Connect a new WebSocket client"""
        await self.manager.connect(websocket, client_id, connection_type, session_id)
    
    async def disconnect(self, client_id: str):
        """Disconnect a WebSocket client"""
        self.manager.disconnect(client_id)
    
    async def subscribe_to_companies(self, client_id: str, company_ids: List[int]):
        """Subscribe client to company updates"""
        self.manager.subscribe_to_companies(client_id, company_ids)
    
    async def unsubscribe_from_companies(self, client_id: str, company_ids: List[int]):
        """Unsubscribe client from company updates"""
        self.manager.unsubscribe_from_companies(client_id, company_ids)
    
    async def subscribe_to_metrics(self, client_id: str):
        """Subscribe client to system metrics"""
        # Mock implementation
        logger.info(f"Client {client_id} subscribed to metrics")
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """Broadcast alert to relevant clients"""
        try:
            message = {
                "type": "alert",
                "timestamp": datetime.utcnow().isoformat(),
                "data": alert_data
            }
            
            # Broadcast to alert subscribers
            await self.manager.broadcast_to_type(json.dumps(message), "alerts")
            
            # Broadcast to dashboard subscribers
            await self.manager.broadcast_to_type(json.dumps(message), "dashboard")
            
            # Broadcast to company-specific subscribers
            if "company_id" in alert_data:
                await self.manager.broadcast_to_subscribers(
                    json.dumps(message), 
                    alert_data["company_id"]
                )
            
            logger.info(f"Broadcasted alert for company {alert_data.get('company_id')}")
            
        except Exception as e:
            logger.error(f"Error broadcasting alert: {str(e)}")
    
    async def broadcast_score_update(self, score_data: Dict[str, Any]):
        """Broadcast score update to subscribed clients"""
        try:
            message = {
                "type": "score_update",
                "timestamp": datetime.utcnow().isoformat(),
                "data": score_data
            }
            
            # Broadcast to score subscribers
            await self.manager.broadcast_to_type(json.dumps(message), "scores")
            
            # Broadcast to dashboard subscribers
            await self.manager.broadcast_to_type(json.dumps(message), "dashboard")
            
            # Broadcast to company-specific subscribers
            if "company_id" in score_data:
                await self.manager.broadcast_to_subscribers(
                    json.dumps(message),
                    score_data["company_id"]
                )
            
            logger.info(f"Broadcasted score update for company {score_data.get('company_id')}")
            
        except Exception as e:
            logger.error(f"Error broadcasting score update: {str(e)}")
    
    async def broadcast_system_status(self, status_data: Dict[str, Any]):
        """Broadcast system status update"""
        try:
            message = {
                "type": "system_status",
                "timestamp": datetime.utcnow().isoformat(),
                "data": status_data
            }
            
            # Broadcast to all dashboard subscribers
            await self.manager.broadcast_to_type(json.dumps(message), "dashboard")
            
            logger.info("Broadcasted system status update")
            
        except Exception as e:
            logger.error(f"Error broadcasting system status: {str(e)}")
    
    async def process_chat_message(
        self,
        session_id: str,
        message: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process chat message and return response"""
        try:
            # Mock chat processing - would integrate with ChatService
            response = {
                "type": "chat_response",
                "session_id": session_id,
                "message": f"I understand you're asking about: {message}. Let me help you with that analysis.",
                "timestamp": datetime.utcnow().isoformat(),
                "confidence": 0.85,
                "sources": ["financial_data", "market_analysis"],
                "follow_up_questions": [
                    "Would you like more details about this analysis?",
                    "Should I compare this with other companies?"
                ]
            }
            
            logger.info(f"Processed chat message for session {session_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            return {
                "type": "error",
                "message": "Sorry, I encountered an error processing your message.",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return self.manager.get_connection_count()
    
    def get_connections_by_type(self) -> Dict[str, int]:
        """Get connection count by type"""
        return self.manager.get_connections_by_type()
    
    def get_active_subscriptions(self) -> Dict[str, int]:
        """Get active subscription counts"""
        return self.manager.get_active_subscriptions()
