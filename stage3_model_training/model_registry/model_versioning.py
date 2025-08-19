"""
Model versioning system for Stage 3.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import json
import os
import shutil
from datetime import datetime
import hashlib
import joblib

logger = logging.getLogger(__name__)

class ModelVersionManager:
    """Manage model versions and lifecycle"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry_path = config.get('registry_path', './model_registry')
        self.versions = {}
        self._ensure_registry_structure()
        
    def _ensure_registry_structure(self):
        """Create registry directory structure"""
        os.makedirs(self.registry_path, exist_ok=True)
        os.makedirs(os.path.join(self.registry_path, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.registry_path, 'metadata'), exist_ok=True)
        os.makedirs(os.path.join(self.registry_path, 'artifacts'), exist_ok=True)
        
    def register_model(self, model: Any, model_name: str, version: str = None,
                      metadata: Dict[str, Any] = None, artifacts: Dict[str, Any] = None) -> str:
        """Register a new model version"""
        
        if version is None:
            version = self._generate_version(model_name)
        
        model_id = f"{model_name}_v{version}"
        
        # Create model directory
        model_dir = os.path.join(self.registry_path, 'models', model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'model.pkl')
        joblib.dump(model, model_path)
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(model_path)
        
        # Create version metadata
        version_metadata = {
            'model_id': model_id,
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'model_hash': model_hash,
            'model_path': model_path,
            'model_type': type(model).__name__,
            'status': 'registered',
            'metadata': metadata or {},
            'artifacts': artifacts or {}
        }
        
        # Save metadata
        metadata_path = os.path.join(self.registry_path, 'metadata', f"{model_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(version_metadata, f, indent=2)
        
        # Save artifacts
        if artifacts:
            self._save_artifacts(model_id, artifacts)
        
        # Update registry
        self.versions[model_id] = version_metadata
        
        logger.info(f"Registered model {model_id} with hash {model_hash}")
        return model_id
    
    def _generate_version(self, model_name: str) -> str:
        """Generate next version number"""
        existing_versions = [
            v['version'] for v in self.versions.values() 
            if v['model_name'] == model_name
        ]
        
        if not existing_versions:
            return "1.0.0"
        
        # Parse semantic versions and increment
        version_numbers = []
        for v in existing_versions:
            try:
                parts = v.split('.')
                version_numbers.append([int(p) for p in parts])
            except:
                continue
        
        if not version_numbers:
            return "1.0.0"
        
        # Get latest version and increment patch
        latest = max(version_numbers)
        latest[2] += 1  # Increment patch version
        
        return '.'.join(map(str, latest))
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA256 hash of model file"""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _save_artifacts(self, model_id: str, artifacts: Dict[str, Any]):
        """Save model artifacts"""
        artifacts_dir = os.path.join(self.registry_path, 'artifacts', model_id)
        os.makedirs(artifacts_dir, exist_ok=True)
        
        for artifact_name, artifact_data in artifacts.items():
            artifact_path = os.path.join(artifacts_dir, f"{artifact_name}.pkl")
            joblib.dump(artifact_data, artifact_path)
    
    def load_model(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """Load model and metadata by ID"""
        
        if model_id not in self.versions:
            # Try to load from disk
            metadata_path = os.path.join(self.registry_path, 'metadata', f"{model_id}.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.versions[model_id] = metadata
            else:
                raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.versions[model_id]
        model_path = metadata['model_path']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        return model, metadata
    
    def load_latest_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Load latest version of a model"""
        
        model_versions = [
            (v['version'], v['model_id']) for v in self.versions.values()
            if v['model_name'] == model_name
        ]
        
        if not model_versions:
            raise ValueError(f"No versions found for model {model_name}")
        
        # Sort versions and get latest
        model_versions.sort(key=lambda x: [int(p) for p in x[0].split('.')])
        latest_model_id = model_versions[-1][1]
        
        return self.load_model(latest_model_id)
    
    def list_models(self, model_name: str = None) -> List[Dict[str, Any]]:
        """List all models or models for specific name"""
        
        # Load all metadata from disk
        self._load_all_metadata()
        
        if model_name:
            return [v for v in self.versions.values() if v['model_name'] == model_name]
        else:
            return list(self.versions.values())
    
    def _load_all_metadata(self):
        """Load all metadata files from disk"""
        metadata_dir = os.path.join(self.registry_path, 'metadata')
        
        if not os.path.exists(metadata_dir):
            return
        
        for filename in os.listdir(metadata_dir):
            if filename.endswith('.json'):
                metadata_path = os.path.join(metadata_dir, filename)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.versions[metadata['model_id']] = metadata
    
    def update_model_status(self, model_id: str, status: str, notes: str = None):
        """Update model status"""
        
        if model_id not in self.versions:
            raise ValueError(f"Model {model_id} not found")
        
        self.versions[model_id]['status'] = status
        self.versions[model_id]['status_updated'] = datetime.now().isoformat()
        
        if notes:
            self.versions[model_id]['status_notes'] = notes
        
        # Save updated metadata
        metadata_path = os.path.join(self.registry_path, 'metadata', f"{model_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.versions[model_id], f, indent=2)
        
        logger.info(f"Updated status of {model_id} to {status}")
    
    def promote_model(self, model_id: str, environment: str):
        """Promote model to environment (staging, production)"""
        
        valid_environments = ['staging', 'production']
        if environment not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        
        self.update_model_status(model_id, f"promoted_to_{environment}")
        
        # Create promotion record
        promotion_record = {
            'model_id': model_id,
            'environment': environment,
            'promoted_at': datetime.now().isoformat(),
            'promoted_by': 'system'  # Could be user ID in real system
        }
        
        # Save promotion record
        promotions_dir = os.path.join(self.registry_path, 'promotions')
        os.makedirs(promotions_dir, exist_ok=True)
        
        promotion_file = os.path.join(promotions_dir, f"{environment}_current.json")
        with open(promotion_file, 'w') as f:
            json.dump(promotion_record, f, indent=2)
        
        logger.info(f"Promoted {model_id} to {environment}")
    
    def get_production_model(self, model_name: str = None) -> Tuple[Any, Dict[str, Any]]:
        """Get current production model"""
        
        promotion_file = os.path.join(self.registry_path, 'promotions', 'production_current.json')
        
        if not os.path.exists(promotion_file):
            raise ValueError("No model currently in production")
        
        with open(promotion_file, 'r') as f:
            promotion_record = json.load(f)
        
        model_id = promotion_record['model_id']
        
        # Verify model name if specified
        if model_name:
            model, metadata = self.load_model(model_id)
            if metadata['model_name'] != model_name:
                raise ValueError(f"Production model is not of type {model_name}")
            return model, metadata
        
        return self.load_model(model_id)
    
    def rollback_model(self, model_name: str, environment: str = 'production'):
        """Rollback to previous model version"""
        
        # Get promotion history
        promotions_dir = os.path.join(self.registry_path, 'promotions')
        promotion_files = [
            f for f in os.listdir(promotions_dir) 
            if f.startswith(f"{environment}_") and f.endswith('.json')
        ]
        
        if len(promotion_files) < 2:
            raise ValueError("No previous version to rollback to")
        
        # Sort by modification time and get previous
        promotion_files.sort(key=lambda x: os.path.getmtime(os.path.join(promotions_dir, x)))
        previous_file = promotion_files[-2]
        
        with open(os.path.join(promotions_dir, previous_file), 'r') as f:
            previous_promotion = json.load(f)
        
        # Promote previous model
        self.promote_model(previous_promotion['model_id'], environment)
        
        logger.info(f"Rolled back {environment} to {previous_promotion['model_id']}")
    
    def delete_model(self, model_id: str, force: bool = False):
        """Delete model version"""
        
        if model_id not in self.versions:
            raise ValueError(f"Model {model_id} not found")
        
        metadata = self.versions[model_id]
        
        # Check if model is in production
        if metadata.get('status', '').startswith('promoted_to_production') and not force:
            raise ValueError("Cannot delete production model without force=True")
        
        # Remove files
        model_dir = os.path.dirname(metadata['model_path'])
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        
        # Remove metadata
        metadata_path = os.path.join(self.registry_path, 'metadata', f"{model_id}.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # Remove artifacts
        artifacts_dir = os.path.join(self.registry_path, 'artifacts', model_id)
        if os.path.exists(artifacts_dir):
            shutil.rmtree(artifacts_dir)
        
        # Remove from memory
        del self.versions[model_id]
        
        logger.info(f"Deleted model {model_id}")
    
    def compare_models(self, model_id1: str, model_id2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        
        metadata1 = self.versions.get(model_id1)
        metadata2 = self.versions.get(model_id2)
        
        if not metadata1 or not metadata2:
            raise ValueError("One or both models not found")
        
        comparison = {
            'model1': {
                'id': model_id1,
                'version': metadata1['version'],
                'timestamp': metadata1['timestamp'],
                'hash': metadata1['model_hash']
            },
            'model2': {
                'id': model_id2,
                'version': metadata2['version'],
                'timestamp': metadata2['timestamp'],
                'hash': metadata2['model_hash']
            },
            'same_model': metadata1['model_hash'] == metadata2['model_hash'],
            'metadata_diff': self._compare_metadata(metadata1['metadata'], metadata2['metadata'])
        }
        
        return comparison
    
    def _compare_metadata(self, meta1: Dict[str, Any], meta2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare metadata between models"""
        
        all_keys = set(meta1.keys()) | set(meta2.keys())
        differences = {}
        
        for key in all_keys:
            val1 = meta1.get(key)
            val2 = meta2.get(key)
            
            if val1 != val2:
                differences[key] = {
                    'model1': val1,
                    'model2': val2
                }
        
        return differences
    
    def get_model_lineage(self, model_name: str) -> List[Dict[str, Any]]:
        """Get version history for a model"""
        
        model_versions = [
            v for v in self.versions.values()
            if v['model_name'] == model_name
        ]
        
        # Sort by timestamp
        model_versions.sort(key=lambda x: x['timestamp'])
        
        return model_versions
    
    def cleanup_old_versions(self, model_name: str, keep_latest: int = 5):
        """Clean up old model versions, keeping only the latest N"""
        
        lineage = self.get_model_lineage(model_name)
        
        if len(lineage) <= keep_latest:
            logger.info(f"No cleanup needed for {model_name}")
            return
        
        # Keep latest versions and production models
        to_delete = []
        production_models = set()
        
        # Find production models
        for version in lineage:
            if version.get('status', '').startswith('promoted_to_production'):
                production_models.add(version['model_id'])
        
        # Mark old versions for deletion (excluding production and latest N)
        sorted_versions = sorted(lineage, key=lambda x: x['timestamp'], reverse=True)
        
        for i, version in enumerate(sorted_versions):
            if i >= keep_latest and version['model_id'] not in production_models:
                to_delete.append(version['model_id'])
        
        # Delete old versions
        for model_id in to_delete:
            try:
                self.delete_model(model_id)
                logger.info(f"Cleaned up old version {model_id}")
            except Exception as e:
                logger.error(f"Failed to delete {model_id}: {e}")
        
        logger.info(f"Cleanup completed for {model_name}, deleted {len(to_delete)} versions")
    
    def export_model(self, model_id: str, export_path: str):
        """Export model and metadata to external location"""
        
        model, metadata = self.load_model(model_id)
        
        # Create export directory
        os.makedirs(export_path, exist_ok=True)
        
        # Export model
        model_export_path = os.path.join(export_path, 'model.pkl')
        joblib.dump(model, model_export_path)
        
        # Export metadata
        metadata_export_path = os.path.join(export_path, 'metadata.json')
        with open(metadata_export_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Export artifacts if they exist
        artifacts_dir = os.path.join(self.registry_path, 'artifacts', model_id)
        if os.path.exists(artifacts_dir):
            export_artifacts_dir = os.path.join(export_path, 'artifacts')
            shutil.copytree(artifacts_dir, export_artifacts_dir)
        
        logger.info(f"Exported {model_id} to {export_path}")
    
    def import_model(self, import_path: str, model_name: str = None) -> str:
        """Import model from external location"""
        
        # Load metadata
        metadata_path = os.path.join(import_path, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Metadata file not found in import path")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        model_path = os.path.join(import_path, 'model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found in import path")
        
        model = joblib.load(model_path)
        
        # Load artifacts if they exist
        artifacts = {}
        artifacts_dir = os.path.join(import_path, 'artifacts')
        if os.path.exists(artifacts_dir):
            for artifact_file in os.listdir(artifacts_dir):
                if artifact_file.endswith('.pkl'):
                    artifact_name = artifact_file[:-4]  # Remove .pkl extension
                    artifact_path = os.path.join(artifacts_dir, artifact_file)
                    artifacts[artifact_name] = joblib.load(artifact_path)
        
        # Register imported model
        imported_model_name = model_name or metadata.get('model_name', 'imported_model')
        model_id = self.register_model(
            model=model,
            model_name=imported_model_name,
            metadata=metadata.get('metadata', {}),
            artifacts=artifacts
        )
        
        logger.info(f"Imported model as {model_id}")
        return model_id
