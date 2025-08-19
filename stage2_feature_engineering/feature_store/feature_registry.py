"""
Feature registry for managing feature metadata and lineage.
Tracks feature definitions, transformations, and dependencies.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import json

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    DATETIME = "datetime"

class FeatureStatus(Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    ARCHIVED = "archived"

@dataclass
class FeatureDefinition:
    """Feature definition with metadata"""
    name: str
    feature_type: FeatureType
    description: str
    source_table: str
    transformation: str
    dependencies: List[str]
    tags: List[str]
    owner: str
    status: FeatureStatus
    created_date: datetime
    last_updated: datetime
    version: str = "1.0"
    data_quality_checks: List[str] = None
    business_logic: str = ""
    expected_range: Optional[Dict[str, float]] = None

class FeatureRegistry:
    """Registry for managing feature definitions and metadata"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.features = {}
        self.feature_groups = {}
        self.dependencies_graph = {}
        
    def register_feature(self, feature_def: FeatureDefinition) -> bool:
        """Register a new feature"""
        try:
            if feature_def.data_quality_checks is None:
                feature_def.data_quality_checks = []
            
            self.features[feature_def.name] = feature_def
            self._update_dependencies_graph(feature_def)
            
            logger.info(f"Registered feature: {feature_def.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering feature {feature_def.name}: {e}")
            return False
    
    def get_feature(self, feature_name: str) -> Optional[FeatureDefinition]:
        """Get feature definition by name"""
        return self.features.get(feature_name)
    
    def list_features(self, status: FeatureStatus = None, 
                     feature_type: FeatureType = None,
                     tags: List[str] = None) -> List[FeatureDefinition]:
        """List features with optional filters"""
        features = list(self.features.values())
        
        if status:
            features = [f for f in features if f.status == status]
        
        if feature_type:
            features = [f for f in features if f.feature_type == feature_type]
        
        if tags:
            features = [f for f in features if any(tag in f.tags for tag in tags)]
        
        return features
    
    def update_feature(self, feature_name: str, updates: Dict[str, Any]) -> bool:
        """Update feature definition"""
        if feature_name not in self.features:
            logger.warning(f"Feature {feature_name} not found for update")
            return False
        
        try:
            feature = self.features[feature_name]
            
            for field, value in updates.items():
                if hasattr(feature, field):
                    setattr(feature, field, value)
            
            feature.last_updated = datetime.now()
            self._update_dependencies_graph(feature)
            
            logger.info(f"Updated feature: {feature_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating feature {feature_name}: {e}")
            return False
    
    def deprecate_feature(self, feature_name: str, reason: str = "") -> bool:
        """Deprecate a feature"""
        return self.update_feature(feature_name, {
            'status': FeatureStatus.DEPRECATED,
            'description': f"{self.features[feature_name].description} [DEPRECATED: {reason}]"
        })
    
    def get_feature_dependencies(self, feature_name: str) -> List[str]:
        """Get direct dependencies of a feature"""
        feature = self.get_feature(feature_name)
        return feature.dependencies if feature else []
    
    def get_dependent_features(self, feature_name: str) -> List[str]:
        """Get features that depend on this feature"""
        dependents = []
        for name, feature in self.features.items():
            if feature_name in feature.dependencies:
                dependents.append(name)
        return dependents
    
    def get_feature_lineage(self, feature_name: str) -> Dict[str, Any]:
        """Get complete lineage of a feature"""
        if feature_name not in self.features:
            return {}
        
        def get_upstream_lineage(name: str, visited: Set[str] = None) -> Dict[str, Any]:
            if visited is None:
                visited = set()
            
            if name in visited:
                return {"circular_dependency": True}
            
            visited.add(name)
            feature = self.features.get(name)
            if not feature:
                return {}
            
            lineage = {
                "name": name,
                "type": feature.feature_type.value,
                "source": feature.source_table,
                "transformation": feature.transformation,
                "dependencies": []
            }
            
            for dep in feature.dependencies:
                dep_lineage = get_upstream_lineage(dep, visited.copy())
                if dep_lineage:
                    lineage["dependencies"].append(dep_lineage)
            
            return lineage
        
        return get_upstream_lineage(feature_name)
    
    def validate_dependencies(self) -> Dict[str, List[str]]:
        """Validate feature dependencies"""
        issues = {}
        
        for feature_name, feature in self.features.items():
            feature_issues = []
            
            # Check for missing dependencies
            for dep in feature.dependencies:
                if dep not in self.features:
                    feature_issues.append(f"Missing dependency: {dep}")
            
            # Check for circular dependencies
            if self._has_circular_dependency(feature_name):
                feature_issues.append("Circular dependency detected")
            
            if feature_issues:
                issues[feature_name] = feature_issues
        
        return issues
    
    def _has_circular_dependency(self, feature_name: str, visited: Set[str] = None) -> bool:
        """Check for circular dependencies"""
        if visited is None:
            visited = set()
        
        if feature_name in visited:
            return True
        
        visited.add(feature_name)
        feature = self.features.get(feature_name)
        
        if not feature:
            return False
        
        for dep in feature.dependencies:
            if self._has_circular_dependency(dep, visited.copy()):
                return True
        
        return False
    
    def _update_dependencies_graph(self, feature: FeatureDefinition):
        """Update internal dependencies graph"""
        self.dependencies_graph[feature.name] = feature.dependencies
    
    def create_feature_group(self, group_name: str, feature_names: List[str], 
                           description: str = "") -> bool:
        """Create a feature group"""
        try:
            # Validate all features exist
            missing_features = [name for name in feature_names if name not in self.features]
            if missing_features:
                logger.error(f"Cannot create group {group_name}: missing features {missing_features}")
                return False
            
            self.feature_groups[group_name] = {
                'features': feature_names,
                'description': description,
                'created_date': datetime.now()
            }
            
            logger.info(f"Created feature group: {group_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating feature group {group_name}: {e}")
            return False
    
    def get_feature_group(self, group_name: str) -> Optional[Dict[str, Any]]:
        """Get feature group"""
        return self.feature_groups.get(group_name)
    
    def list_feature_groups(self) -> List[str]:
        """List all feature groups"""
        return list(self.feature_groups.keys())
    
    def search_features(self, query: str) -> List[FeatureDefinition]:
        """Search features by name, description, or tags"""
        query = query.lower()
        results = []
        
        for feature in self.features.values():
            if (query in feature.name.lower() or
                query in feature.description.lower() or
                any(query in tag.lower() for tag in feature.tags)):
                results.append(feature)
        
        return results
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_features = len(self.features)
        
        stats = {
            'total_features': total_features,
            'by_status': {},
            'by_type': {},
            'by_owner': {},
            'total_groups': len(self.feature_groups),
            'dependency_issues': len(self.validate_dependencies())
        }
        
        for feature in self.features.values():
            # Count by status
            status = feature.status.value
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            
            # Count by type
            ftype = feature.feature_type.value
            stats['by_type'][ftype] = stats['by_type'].get(ftype, 0) + 1
            
            # Count by owner
            owner = feature.owner
            stats['by_owner'][owner] = stats['by_owner'].get(owner, 0) + 1
        
        return stats
    
    def export_registry(self, filepath: str) -> bool:
        """Export registry to JSON file"""
        try:
            export_data = {
                'features': {},
                'feature_groups': self.feature_groups,
                'export_date': datetime.now().isoformat()
            }
            
            # Convert features to serializable format
            for name, feature in self.features.items():
                feature_dict = asdict(feature)
                feature_dict['feature_type'] = feature.feature_type.value
                feature_dict['status'] = feature.status.value
                feature_dict['created_date'] = feature.created_date.isoformat()
                feature_dict['last_updated'] = feature.last_updated.isoformat()
                export_data['features'][name] = feature_dict
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Registry exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting registry: {e}")
            return False
    
    def import_registry(self, filepath: str) -> bool:
        """Import registry from JSON file"""
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            # Import features
            for name, feature_dict in import_data.get('features', {}).items():
                feature_dict['feature_type'] = FeatureType(feature_dict['feature_type'])
                feature_dict['status'] = FeatureStatus(feature_dict['status'])
                feature_dict['created_date'] = datetime.fromisoformat(feature_dict['created_date'])
                feature_dict['last_updated'] = datetime.fromisoformat(feature_dict['last_updated'])
                
                feature = FeatureDefinition(**feature_dict)
                self.register_feature(feature)
            
            # Import feature groups
            self.feature_groups.update(import_data.get('feature_groups', {}))
            
            logger.info(f"Registry imported from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing registry: {e}")
            return False
    
    def generate_documentation(self) -> str:
        """Generate feature documentation"""
        doc = "# Feature Registry Documentation\n\n"
        
        # Summary
        stats = self.get_feature_statistics()
        doc += f"**Total Features:** {stats['total_features']}\n"
        doc += f"**Feature Groups:** {stats['total_groups']}\n\n"
        
        # Features by status
        doc += "## Features by Status\n"
        for status, count in stats['by_status'].items():
            doc += f"- {status.title()}: {count}\n"
        doc += "\n"
        
        # Feature groups
        if self.feature_groups:
            doc += "## Feature Groups\n"
            for group_name, group_info in self.feature_groups.items():
                doc += f"### {group_name}\n"
                doc += f"{group_info.get('description', 'No description')}\n"
                doc += f"**Features:** {', '.join(group_info['features'])}\n\n"
        
        # Individual features
        doc += "## Feature Definitions\n"
        for feature in sorted(self.features.values(), key=lambda x: x.name):
            doc += f"### {feature.name}\n"
            doc += f"**Type:** {feature.feature_type.value}\n"
            doc += f"**Status:** {feature.status.value}\n"
            doc += f"**Description:** {feature.description}\n"
            doc += f"**Source:** {feature.source_table}\n"
            doc += f"**Owner:** {feature.owner}\n"
            
            if feature.dependencies:
                doc += f"**Dependencies:** {', '.join(feature.dependencies)}\n"
            
            if feature.tags:
                doc += f"**Tags:** {', '.join(feature.tags)}\n"
            
            doc += "\n"
        
        return doc
