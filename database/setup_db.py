"""
Database setup script for Credit Intelligence Platform
Creates database, runs initialization scripts, and sets up connections
"""

import asyncio
import asyncpg
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
import os
from pathlib import Path
import sys
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Database setup and initialization"""
    
    def __init__(self, config_path: str = None):
        # Database connection parameters from environment variables
        self.db_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
            'database': os.getenv('DB_NAME', 'credit_intelligence')
        }
        
        # Admin connection (without database specified)
        self.admin_params = self.db_params.copy()
        self.admin_params['database'] = 'postgres'
        
    def create_database(self):
        """Create the database if it doesn't exist"""
        try:
            # Connect to PostgreSQL server (not specific database)
            conn = psycopg2.connect(**self.admin_params)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (self.db_params['database'],)
            )
            
            if not cursor.fetchone():
                # Create database
                cursor.execute(f'CREATE DATABASE "{self.db_params["database"]}"')
                logger.info(f"Created database: {self.db_params['database']}")
            else:
                logger.info(f"Database {self.db_params['database']} already exists")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            return False
    
    def run_init_script(self):
        """Run the database initialization script"""
        try:
            # Connect to the specific database
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # Read and execute init script
            init_script_path = Path(__file__).parent / 'init_db.sql'
            
            if not init_script_path.exists():
                logger.error(f"Init script not found: {init_script_path}")
                return False
            
            with open(init_script_path, 'r', encoding='utf-8') as f:
                init_script = f.read()
            
            # Execute the script
            cursor.execute(init_script)
            conn.commit()
            
            logger.info("Database initialization script executed successfully")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error running init script: {e}")
            return False
    
    async def test_async_connection(self):
        """Test async database connection"""
        try:
            # Test connection using asyncpg.connect with individual parameters
            conn = await asyncpg.connect(
                host=self.db_params['host'],
                port=self.db_params['port'],
                user=self.db_params['user'],
                password=self.db_params['password'],
                database=self.db_params['database']
            )
            
            # Test query
            result = await conn.fetchval("SELECT COUNT(*) FROM companies")
            logger.info(f"Async connection test successful. Found {result} companies in database.")
            
            await conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Async connection test failed: {e}")
            return False

    
    def create_user_and_permissions(self):
        """Create application user and set permissions"""
        try:
            conn = psycopg2.connect(**self.admin_params)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            app_user = os.getenv('DB_APP_USER', 'credit_intelligence_user')
            app_password = os.getenv('DB_APP_PASSWORD', 'secure_password_123')
            
            # Check if user exists
            cursor.execute(
                "SELECT 1 FROM pg_roles WHERE rolname = %s",
                (app_user,)
            )
            
            if not cursor.fetchone():
                # Create user
                cursor.execute(f'CREATE USER "{app_user}" WITH PASSWORD %s', (app_password,))
                logger.info(f"Created database user: {app_user}")
            else:
                logger.info(f"Database user {app_user} already exists")
            
            cursor.close()
            conn.close()
            
            # Grant permissions on the application database
            conn = psycopg2.connect(**self.db_params)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Grant permissions
            cursor.execute(f'GRANT ALL PRIVILEGES ON DATABASE "{self.db_params["database"]}" TO "{app_user}"')
            cursor.execute(f'GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO "{app_user}"')
            cursor.execute(f'GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO "{app_user}"')
            cursor.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO "{app_user}"')
            cursor.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO "{app_user}"')
            
            logger.info(f"Granted permissions to user: {app_user}")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error creating user and permissions: {e}")
            return False
    
    def setup_complete_database(self):
        """Complete database setup process"""
        logger.info("Starting database setup...")
        
        steps = [
            ("Creating database", self.create_database),
            ("Running initialization script", self.run_init_script),
            ("Creating application user", self.create_user_and_permissions),
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            if not step_func():
                logger.error(f"Failed at step: {step_name}")
                return False
            logger.info(f"✅ Completed: {step_name}")
        
        logger.info("🎉 Database setup completed successfully!")
        return True
    
    async def verify_setup(self):
        """Verify database setup is working"""
        logger.info("Verifying database setup...")
        
        # Test async connection
        if not await self.test_async_connection():
            return False
        
        try:
            # Test basic operations
            conn = await asyncpg.connect(
                host=self.db_params['host'],
                port=self.db_params['port'],
                user=self.db_params['user'],
                password=self.db_params['password'],
                database=self.db_params['database']
            )
            
            # Test inserting and querying data
            test_ticker = 'TEST'
            await conn.execute(
                "INSERT INTO companies (ticker, name, sector) VALUES ($1, $2, $3) ON CONFLICT (ticker) DO NOTHING",
                test_ticker, 'Test Company', 'Technology'
            )
            
            result = await conn.fetchrow("SELECT * FROM companies WHERE ticker = $1", test_ticker)
            if result:
                logger.info("✅ Database read/write operations working")
            
            # Clean up test data
            await conn.execute("DELETE FROM companies WHERE ticker = $1", test_ticker)
            
            await conn.close()
            
            logger.info("🎉 Database verification completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Database verification failed: {e}")
            return False

def main():
    """Main setup function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if PostgreSQL is running using config values
    try:
        import psycopg2
        test_conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'password'),
            database='postgres'
        )
        test_conn.close()
        logger.info("✅ PostgreSQL server is running")
    except Exception as e:
        logger.error(f"❌ Cannot connect to PostgreSQL server: {e}")
        logger.error("Please ensure PostgreSQL is installed and running")
        logger.error("Check your .env file for correct database credentials")
        return False
    
    # Setup database
    db_setup = DatabaseSetup()
    
    if not db_setup.setup_complete_database():
        logger.error("❌ Database setup failed")
        return False
    
    # Verify setup
    async def verify():
        return await db_setup.verify_setup()
    
    if not asyncio.run(verify()):
        logger.error("❌ Database verification failed")
        return False
    
    logger.info("🚀 Database is ready for the Credit Intelligence Platform!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
