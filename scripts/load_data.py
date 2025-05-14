"""
Script to load and index data from a CSV file.
"""

import os
import sys
import asyncio
import argparse
import pandas as pd
import logging
from app.services.azure_openai import AzureOpenAIService
from app.services.elasticsearch_service import ElasticsearchService
from app.services.vector_service import VectorService
from app.core.environment import get_os_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def load_data(csv_path: str, force_recreate: bool = False):
    """
    Load and index data from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        force_recreate: Whether to force recreate the index
    """
    try:
        logger.info(f"Loading data from {csv_path}")
        
        # Create services
        azure_service = AzureOpenAIService()
        es_service = ElasticsearchService()
        
        # Connect to Elasticsearch
        await es_service.connect()
        
        # Create index
        await es_service.create_index(force=force_recreate)
        
        # Create vector service
        vector_service = VectorService(azure_service, es_service)
        
        # Index data
        await vector_service.index_data_from_csv(csv_path)
        
        logger.info("Data loading completed successfully")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    finally:
        # Close Elasticsearch connection
        if es_service and hasattr(es_service, 'client') and es_service.client:
            await es_service.close()

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Load and index data from a CSV file")
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--force", action="store_true", help="Force recreate the Elasticsearch index")
    args = parser.parse_args()
    
    # Load environment variables
    env = get_os_env()
    
    # Run the async function
    asyncio.run(load_data(args.csv, args.force))

if __name__ == "__main__":
    main()