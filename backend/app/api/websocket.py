"""
WebSocket endpoints for real-time communication
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import List, Dict, Any
import json
import asyncio
import logging
from datetime import datetime

from app.services.websocket_service import WebSocketService
from app.utils.auth import get_current_user_ws

router = APIRouter()
logger = logging.getLogger(__name__)

# Global WebSocket manager
websocket_service = WebSocketService()

@router.websocket("/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alert notifications"""
    await websocket.accept()
    client_id = f"alerts_{datetime.utcnow().timestamp()}"
    
    try:
        await websocket_service.connect(websocket, client_id, "alerts")
        
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle client messages (e.g., subscribe to specific companies)
            if message.get("type") == "subscribe":
                company_ids = message.get("company_ids", [])
                await websocket_service.subscribe_to_companies(client_id, company_ids)
                
            elif message.get("type") == "unsubscribe":
                company_ids = message.get("company_ids", [])
                await websocket_service.unsubscribe_from_companies(client_id, company_ids)
                
    except WebSocketDisconnect:
        await websocket_service.disconnect(client_id)
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        await websocket_service.disconnect(client_id)

@router.websocket("/scores")
async def websocket_scores(websocket: WebSocket):
    """WebSocket endpoint for real-time score updates"""
    await websocket.accept()
    client_id = f"scores_{datetime.utcnow().timestamp()}"
    
    try:
        await websocket_service.connect(websocket, client_id, "scores")
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                company_ids = message.get("company_ids", [])
                await websocket_service.subscribe_to_companies(client_id, company_ids)
                
    except WebSocketDisconnect:
        await websocket_service.disconnect(client_id)
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        await websocket_service.disconnect(client_id)

@router.websocket("/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket endpoint for dashboard real-time updates"""
    await websocket.accept()
    client_id = f"dashboard_{datetime.utcnow().timestamp()}"
    
    try:
        await websocket_service.connect(websocket, client_id, "dashboard")
        
        # Send initial dashboard data
        initial_data = {
            "type": "initial_data",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "active_alerts": 23,
                "companies_monitored": 1250,
                "system_status": "operational"
            }
        }
        await websocket.send_text(json.dumps(initial_data))
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle dashboard-specific subscriptions
            if message.get("type") == "subscribe_metrics":
                await websocket_service.subscribe_to_metrics(client_id)
                
    except WebSocketDisconnect:
        await websocket_service.disconnect(client_id)
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        await websocket_service.disconnect(client_id)

@router.websocket("/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    client_id = f"chat_{session_id}_{datetime.utcnow().timestamp()}"
    
    try:
        await websocket_service.connect(websocket, client_id, "chat", session_id)
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "message":
                # Process chat message and send response
                response = await websocket_service.process_chat_message(
                    session_id=session_id,
                    message=message.get("content"),
                    user_id=message.get("user_id")
                )
                await websocket.send_text(json.dumps(response))
                
    except WebSocketDisconnect:
        await websocket_service.disconnect(client_id)
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        await websocket_service.disconnect(client_id)

# HTTP endpoints for WebSocket management
@router.get("/connections")
async def get_active_connections():
    """Get active WebSocket connections"""
    return {
        "total_connections": websocket_service.get_connection_count(),
        "connections_by_type": websocket_service.get_connections_by_type(),
        "active_subscriptions": websocket_service.get_active_subscriptions()
    }

@router.post("/broadcast/alert")
async def broadcast_alert(alert_data: dict):
    """Broadcast alert to all connected clients"""
    try:
        await websocket_service.broadcast_alert(alert_data)
        return {"message": "Alert broadcasted successfully"}
    except Exception as e:
        logger.error(f"Error broadcasting alert: {str(e)}")
        return {"error": "Failed to broadcast alert"}

@router.post("/broadcast/score-update")
async def broadcast_score_update(score_data: dict):
    """Broadcast score update to subscribed clients"""
    try:
        await websocket_service.broadcast_score_update(score_data)
        return {"message": "Score update broadcasted successfully"}
    except Exception as e:
        logger.error(f"Error broadcasting score update: {str(e)}")
        return {"error": "Failed to broadcast score update"}

@router.post("/broadcast/system-status")
async def broadcast_system_status(status_data: dict):
    """Broadcast system status update"""
    try:
        await websocket_service.broadcast_system_status(status_data)
        return {"message": "System status broadcasted successfully"}
    except Exception as e:
        logger.error(f"Error broadcasting system status: {str(e)}")
        return {"error": "Failed to broadcast system status"}
