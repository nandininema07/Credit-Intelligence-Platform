"""
Quick test script to verify database connection using .env variables
"""

import os
import asyncio
import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_connection():
    """Test database connection"""
    try:
        print("Testing database connection...")
        print(f"DB_HOST: {os.getenv('DB_HOST', 'localhost')}")
        print(f"DB_PORT: {os.getenv('DB_PORT', '5432')}")
        print(f"DB_NAME: {os.getenv('DB_NAME', 'credit_intelligence')}")
        print(f"DB_USER: {os.getenv('DB_USER', 'postgres')}")
        print(f"DB_PASSWORD: {'*' * len(os.getenv('DB_PASSWORD', ''))}")
        
        # Test connection
        conn = await asyncpg.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', 'credit_intelligence')
        )
        
        # Test query
        result = await conn.fetchval("SELECT COUNT(*) FROM companies")
        print(f"✅ Connection successful! Found {result} companies")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_connection())
    if success:
        print("🎉 Database connection is working!")
    else:
        print("💥 Database connection failed - check your .env file")
