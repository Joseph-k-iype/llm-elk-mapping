#!/usr/bin/env python
"""
Enhanced CSV Loading Script with Reliable Elasticsearch Indexing.

This script loads business terms from a CSV file, generates embeddings using Azure OpenAI,
and indexes them in Elasticsearch with improved reliability for large datasets.
"""

import os
import sys
import time
import logging
import argparse
import asyncio
import json
import traceback
import pandas as pd
from typing import List, Dict, Any
from app.services.azure_openai import AzureOpenAIService
from app.services.elasticsearch_service import ElasticsearchService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('load_data.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

async def load_csv_data(csv_path: str, 
                       batch_size: int = 3, 
                       es_batch_size: int = 25, 
                       max_retries: int = 3,
                       skip_embeddings: bool = False,
                       skip_indexing: bool = False,
                       start_row: int = 0,
                       end_row: int = None):
    """
    Load data from CSV, generate embeddings, and index in Elasticsearch.
    
    Args:
        csv_path: Path to the CSV file
        batch_size: Size of batches for embedding generation
        es_batch_size: Size of batches for Elasticsearch indexing
        max_retries: Maximum number of retries for API calls
        skip_embeddings: Skip embedding generation (use for testing indexing only)
        skip_indexing: Skip indexing (use for testing embedding generation only)
        start_row: Starting row index for processing (useful for resuming failed jobs)
        end_row: Ending row index for processing (useful for chunking large files)
    """
    try:
        logger.info(f"Loading CSV data from {csv_path}...")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Apply row filtering if specified
        if end_row is not None:
            df = df.iloc[start_row:end_row]
        elif start_row > 0:
            df = df.iloc[start_row:]
            
        logger.info(f"Processing rows {start_row} to {start_row + len(df) - 1}")
        
        # Check if columns exist and rename if needed
        column_mapping = {
            'PBT_NAME': 'pbt_name',
            'PBT_DEFINITION': 'pbt_definition',
            'CDM': 'cdm'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_cols = ['pbt_name', 'pbt_definition']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
        # Ensure id column exists
        if 'id' not in df.columns:
            df['id'] = df.index.astype(str)
        
        # Initialize services
        logger.info("Initializing Azure OpenAI service...")
        azure_service = AzureOpenAIService()
        
        logger.info("Initializing Elasticsearch service...")
        es_service = ElasticsearchService()
        await es_service.connect()
        
        # Check if index exists, create if not
        try:
            logger.info("Checking if Elasticsearch index exists...")
            exists = await es_service.client.indices.exists(index=es_service.index_name)
            if not exists:
                logger.info(f"Creating Elasticsearch index '{es_service.index_name}'...")
                await es_service.create_index(force=False)
            else:
                logger.info(f"Elasticsearch index '{es_service.index_name}' already exists")
        except Exception as e:
            logger.error(f"Error checking/creating Elasticsearch index: {e}")
            raise
        
        # Generate texts for embedding and prepare documents
        logger.info("Preparing texts for embedding...")
        documents = []
        texts = []
        
        for idx, row in df.iterrows():
            # Create document with required fields
            doc = {
                "id": str(row.id),
                "pbt_name": row.pbt_name,
                "pbt_definition": row.pbt_definition,
            }
            
            # Add optional fields if they exist
            if 'cdm' in row and pd.notna(row.cdm):
                doc["cdm"] = row.cdm
                
            documents.append(doc)
            
            # Combine name and definition for better embeddings
            text = f"{row.pbt_name} {row.pbt_definition}"
            texts.append(text)
        
        # Generate embeddings if not skipped
        all_embeddings = []
        
        if not skip_embeddings:
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            start_time = time.time()
            all_embeddings = await azure_service.generate_embeddings(texts)
            embedding_time = time.time() - start_time
            logger.info(f"Generated {len(all_embeddings)} embeddings in {embedding_time:.2f}s")
            
            # Verify all embeddings were generated
            if len(all_embeddings) != len(texts):
                logger.warning(f"Warning: Generated {len(all_embeddings)} embeddings but expected {len(texts)}")
                
            # Save embeddings to file as backup
            embeddings_file = f"{os.path.splitext(csv_path)[0]}_embeddings.json"
            try:
                with open(embeddings_file, 'w') as f:
                    json.dump(all_embeddings, f)
                logger.info(f"Saved embeddings to {embeddings_file}")
            except Exception as e:
                logger.warning(f"Could not save embeddings to file: {e}")
        else:
            logger.info("Skipping embedding generation as requested")
            # Use zeros as placeholder embeddings
            all_embeddings = [[0.0] * 1536 for _ in range(len(texts))]
        
        # Add embeddings to documents
        logger.info("Adding embeddings to documents...")
        for i, doc in enumerate(documents):
            if i < len(all_embeddings):
                doc["embedding"] = all_embeddings[i]
        
        # Skip indexing if requested
        if skip_indexing:
            logger.info("Skipping indexing as requested")
            return
        
        # Index documents in Elasticsearch with smaller batches
        logger.info(f"Indexing {len(documents)} documents to Elasticsearch in batches of {es_batch_size}...")
        success_count = 0
        error_count = 0
        
        # Process in smaller batches
        for i in range(0, len(documents), es_batch_size):
            batch = documents[i:i+es_batch_size]
            batch_num = i // es_batch_size + 1
            total_batches = (len(documents) - 1) // es_batch_size + 1
            
            logger.info(f"Indexing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            
            # Try indexing with retries
            indexed = False
            for retry in range(max_retries):
                try:
                    # Prepare bulk operations
                    operations = []
                    for doc in batch:
                        operations.append({"index": {"_index": es_service.index_name, "_id": doc["id"]}})
                        operations.append(doc)
                    
                    # Execute bulk operation
                    response = await es_service.client.bulk(operations=operations, refresh=True)
                    
                    # Check for errors
                    if response.get("errors", False):
                        error_items = [item for item in response.get("items", []) if item.get("index", {}).get("error")]
                        if error_items:
                            errors = [item["index"]["error"] for item in error_items]
                            logger.error(f"Bulk indexing batch {batch_num} had {len(error_items)} errors: {errors[:3]}")
                            # Continue with retry
                        else:
                            logger.warning(f"Bulk indexing reported errors but no error details found")
                            indexed = True
                            success_count += len(batch)
                            break
                    else:
                        logger.info(f"Successfully indexed batch {batch_num}/{total_batches}")
                        indexed = True
                        success_count += len(batch)
                        break
                except Exception as e:
                    logger.error(f"Error indexing batch {batch_num} (attempt {retry+1}): {e}")
                    logger.error(traceback.format_exc())
                    
                    if retry == max_retries - 1:  # Last retry
                        logger.error(f"Failed to index batch {batch_num} after {max_retries} attempts")
                        error_count += len(batch)
                    else:
                        logger.info(f"Retrying batch {batch_num} in {(retry+1)*2} seconds...")
                        time.sleep((retry+1) * 2)  # Increasing backoff
            
            if not indexed:
                error_count += len(batch)
            
            # Add delay between batches
            if i + es_batch_size < len(documents):
                time.sleep(1.0)  # 1 second delay between batches
                
                # Refresh connection every 10 batches to avoid timeouts
                if batch_num % 10 == 0:
                    try:
                        logger.info("Reconnecting to Elasticsearch to avoid timeout...")
                        await es_service.connect()
                    except Exception as conn_error:
                        logger.error(f"Error reconnecting to Elasticsearch: {conn_error}")
        
        logger.info(f"Indexing completed: {success_count} documents indexed successfully, {error_count} failed")
        
        # Final status
        if error_count > 0:
            logger.warning(f"WARNING: {error_count} documents failed to index")
            logger.info(f"To retry failed documents, you might need to run again with different batch sizes")
        else:
            logger.info("All documents were indexed successfully!")
        
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Clean up
        if 'es_service' in locals() and es_service:
            await es_service.close()
            logger.info("Elasticsearch connection closed")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Load business terms from CSV and generate embeddings')
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--batch-size', type=int, default=3, help='Batch size for embedding generation')
    parser.add_argument('--es-batch-size', type=int, default=25, help='Batch size for Elasticsearch indexing')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries for API calls')
    parser.add_argument('--skip-embeddings', action='store_true', help='Skip embedding generation')
    parser.add_argument('--skip-indexing', action='store_true', help='Skip Elasticsearch indexing')
    parser.add_argument('--start-row', type=int, default=0, help='Starting row index')
    parser.add_argument('--end-row', type=int, default=None, help='Ending row index')
    args = parser.parse_args()
    
    await load_csv_data(
        args.csv, 
        args.batch_size, 
        args.es_batch_size,
        args.max_retries,
        args.skip_embeddings,
        args.skip_indexing,
        args.start_row,
        args.end_row
    )

if __name__ == "__main__":
    asyncio.run(main())