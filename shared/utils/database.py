"""
Shared database utilities for the Credit Intelligence Platform.
Provides common database operations and connection management.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
import asyncpg
import pandas as pd
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Shared database connection and utility manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool = None
        self.connection_string = self._build_connection_string()
        
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string"""
        host = self.config.get('postgres_host', 'localhost')
        port = self.config.get('postgres_port', 5432)
        database = self.config.get('postgres_database', 'credit_intelligence')
        user = self.config.get('postgres_user', 'postgres')
        password = self.config.get('postgres_password', '')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    async def initialize(self, min_connections: int = 5, max_connections: int = 20):
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=min_connections,
                max_size=max_connections,
                command_timeout=60
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute query and return results"""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            try:
                rows = await connection.fetch(query, *args)
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise
    
    async def execute_command(self, command: str, *args) -> str:
        """Execute command (INSERT, UPDATE, DELETE)"""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            try:
                result = await connection.execute(command, *args)
                return result
            except Exception as e:
                logger.error(f"Command execution failed: {e}")
                raise
    
    async def bulk_insert(self, table: str, data: List[Dict[str, Any]]) -> int:
        """Bulk insert data into table"""
        if not data:
            return 0
        
        if not self.pool:
            await self.initialize()
        
        # Prepare columns and values
        columns = list(data[0].keys())
        placeholders = ', '.join([f'${i+1}' for i in range(len(columns))])
        
        query = f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES ({placeholders})
        """
        
        async with self.pool.acquire() as connection:
            try:
                async with connection.transaction():
                    for row in data:
                        values = [row[col] for col in columns]
                        await connection.execute(query, *values)
                
                logger.info(f"Bulk inserted {len(data)} rows into {table}")
                return len(data)
                
            except Exception as e:
                logger.error(f"Bulk insert failed: {e}")
                raise
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information"""
        query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = $1
        ORDER BY ordinal_position
        """
        
        columns = await self.execute_query(query, table_name)
        
        count_query = f"SELECT COUNT(*) as count FROM {table_name}"
        count_result = await self.execute_query(count_query)
        
        return {
            'table_name': table_name,
            'columns': columns,
            'row_count': count_result[0]['count'] if count_result else 0
        }
    
    async def create_index(self, table: str, columns: List[str], index_name: str = None):
        """Create database index"""
        if not index_name:
            index_name = f"idx_{table}_{'_'.join(columns)}"
        
        column_list = ', '.join(columns)
        query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} ({column_list})"
        
        await self.execute_command(query)
        logger.info(f"Created index {index_name} on {table}")
    
    async def optimize_table(self, table: str):
        """Optimize table performance"""
        # Analyze table for query planner
        await self.execute_command(f"ANALYZE {table}")
        
        # Vacuum table to reclaim space
        await self.execute_command(f"VACUUM {table}")
        
        logger.info(f"Optimized table {table}")
    
    async def backup_table(self, table: str, backup_path: str):
        """Backup table to file"""
        query = f"SELECT * FROM {table}"
        data = await self.execute_query(query)
        
        df = pd.DataFrame(data)
        df.to_csv(backup_path, index=False)
        
        logger.info(f"Backed up {table} to {backup_path}")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        # Table sizes
        size_query = """
        SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
            pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
        FROM pg_tables 
        WHERE schemaname = 'public'
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """
        
        table_sizes = await self.execute_query(size_query)
        
        # Connection stats
        conn_query = """
        SELECT 
            state,
            COUNT(*) as count
        FROM pg_stat_activity 
        WHERE datname = current_database()
        GROUP BY state
        """
        
        connections = await self.execute_query(conn_query)
        
        return {
            'table_sizes': table_sizes,
            'connections': connections,
            'pool_size': self.pool.get_size() if self.pool else 0,
            'timestamp': datetime.now().isoformat()
        }

class QueryBuilder:
    """SQL query builder utility"""
    
    @staticmethod
    def select(table: str, columns: List[str] = None, where: Dict[str, Any] = None,
               order_by: str = None, limit: int = None) -> Tuple[str, List[Any]]:
        """Build SELECT query"""
        # Columns
        if columns:
            column_str = ', '.join(columns)
        else:
            column_str = '*'
        
        query = f"SELECT {column_str} FROM {table}"
        params = []
        
        # WHERE clause
        if where:
            conditions = []
            for i, (column, value) in enumerate(where.items(), 1):
                conditions.append(f"{column} = ${i}")
                params.append(value)
            
            query += f" WHERE {' AND '.join(conditions)}"
        
        # ORDER BY
        if order_by:
            query += f" ORDER BY {order_by}"
        
        # LIMIT
        if limit:
            query += f" LIMIT {limit}"
        
        return query, params
    
    @staticmethod
    def insert(table: str, data: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Build INSERT query"""
        columns = list(data.keys())
        placeholders = [f'${i+1}' for i in range(len(columns))]
        
        query = f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        RETURNING id
        """
        
        params = list(data.values())
        return query, params
    
    @staticmethod
    def update(table: str, data: Dict[str, Any], where: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Build UPDATE query"""
        set_clauses = []
        params = []
        
        # SET clause
        for i, (column, value) in enumerate(data.items(), 1):
            set_clauses.append(f"{column} = ${i}")
            params.append(value)
        
        query = f"UPDATE {table} SET {', '.join(set_clauses)}"
        
        # WHERE clause
        if where:
            conditions = []
            for column, value in where.items():
                conditions.append(f"{column} = ${len(params) + 1}")
                params.append(value)
            
            query += f" WHERE {' AND '.join(conditions)}"
        
        return query, params
    
    @staticmethod
    def delete(table: str, where: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Build DELETE query"""
        query = f"DELETE FROM {table}"
        params = []
        
        if where:
            conditions = []
            for i, (column, value) in enumerate(where.items(), 1):
                conditions.append(f"{column} = ${i}")
                params.append(value)
            
            query += f" WHERE {' AND '.join(conditions)}"
        
        return query, params
