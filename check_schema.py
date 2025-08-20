#!/usr/bin/env python3
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def check_schema():
    conn = await asyncpg.connect(
        host=os.getenv('DB_HOST'),
        port=int(os.getenv('DB_PORT')),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )
    
    # Get all tables
    tables = await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    print('Existing tables:', [row['table_name'] for row in tables])
    
    # Check structure of key tables
    key_tables = ['companies', 'credit_scores', 'alerts', 'features', 'raw_data']
    for table in key_tables:
        if table in [row['table_name'] for row in tables]:
            cols = await conn.fetch("SELECT column_name FROM information_schema.columns WHERE table_name = $1", table)
            print(f'{table} table columns:', [row['column_name'] for row in cols])
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(check_schema())
