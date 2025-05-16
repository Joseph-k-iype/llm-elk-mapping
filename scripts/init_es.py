"""
Enhanced script to initialize Elasticsearch with advanced vector search capabilities.
"""

import os
import sys
import asyncio
import argparse
import logging
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError
from app.config.settings import get_settings
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

async def init_elasticsearch(
    force_recreate=False, 
    vector_dims=3072, 
    vector_similarity="cosine",
    hnsw_m=16, 
    hnsw_ef_construction=100, 
    shards=1, 
    replicas=0
):
    """
    Initialize Elasticsearch with advanced vector search capabilities.
    
    Args:
        force_recreate: Whether to force recreate the index if it exists
        vector_dims: Dimension of embedding vectors
        vector_similarity: Similarity function (cosine, dot_product, l2_norm)
        hnsw_m: Number of bi-directional links in HNSW graph
        hnsw_ef_construction: Controls quality of the graph during construction
        shards: Number of primary shards
        replicas: Number of replicas
    """
    # Get settings
    settings = get_settings()
    
    # Get Elasticsearch connection settings
    hosts = settings.elasticsearch.hosts
    index_name = settings.elasticsearch.index_name
    username = settings.elasticsearch.username
    password = settings.elasticsearch.password
    
    # Extract host URL from the array (using first one)
    host_url = hosts[0] if isinstance(hosts, list) else hosts
    
    # Remove quotes if they exist in the URL string
    if isinstance(host_url, str) and (host_url.startswith('"') or host_url.startswith("'")):
        host_url = host_url.strip('\'"')
    
    # Ensure we're using HTTPS
    if not host_url.startswith("https://"):
        # Replace http:// with https:// or add https:// if no protocol is specified
        if host_url.startswith("http://"):
            host_url = host_url.replace("http://", "https://")
        else:
            host_url = f"https://{host_url}"
    
    logger.info(f"Connecting to Elasticsearch at {host_url}")
    
    # Setup auth if credentials are provided
    auth_params = {}
    if username and password:
        auth_params["basic_auth"] = (username, password)
        logger.info(f"Using basic authentication with username: {username}")
    
    # Create Elasticsearch client with settings that worked for the user
    client = AsyncElasticsearch(
        host_url,  # Use direct URL string with HTTPS
        **auth_params,
        verify_certs=False,  # Disable cert verification for development
        ssl_show_warn=False  # Suppress SSL warnings
    )
    
    try:
        # Check if Elasticsearch is running
        info = await client.info()
        es_version = info["version"]["number"]
        cluster_name = info["cluster_name"]
        logger.info(f"Connected to Elasticsearch version {es_version} on cluster '{cluster_name}'")
        
        # Log vector search configuration
        logger.info(f"Vector search configuration:")
        logger.info(f"  - Vector dimensions: {vector_dims}")
        logger.info(f"  - Similarity function: {vector_similarity}")
        logger.info(f"  - HNSW m parameter: {hnsw_m}")
        logger.info(f"  - HNSW ef_construction: {hnsw_ef_construction}")
        logger.info(f"  - Shards: {shards}")
        logger.info(f"  - Replicas: {replicas}")
        
        # Check if index exists
        index_exists = await client.indices.exists(index=index_name)
        
        # Delete index if it exists and force_recreate is True
        if index_exists and force_recreate:
            logger.info(f"Deleting existing index '{index_name}'")
            await client.indices.delete(index=index_name)
            index_exists = False
        
        # Create index if it doesn't exist
        if not index_exists:
            logger.info(f"Creating index '{index_name}' with advanced vector search settings")
            
            # Define enhanced analyzer configuration
            analyzer_config = {
                "analyzer": {
                    "default": {
                        "type": "standard"
                    },
                    "ngram_analyzer": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "ngram_filter"]
                    }
                },
                "filter": {
                    "ngram_filter": {
                        "type": "ngram",
                        "min_gram": 3,
                        "max_gram": 4
                    }
                }
            }
            
            # Define index settings with KNN capabilities
            index_settings = {
                "settings": {
                    "number_of_shards": shards,
                    "number_of_replicas": replicas,
                    "index.knn": True,  # Enable KNN capabilities
                    "analysis": analyzer_config,
                    "index.mapping.coerce": True,
                    "index.mapping.ignore_malformed": True,
                    # Add memory circuit breaker settings
                    "index.knn.memory_circuit_breaker.enabled": True,
                    "index.knn.memory_circuit_breaker.limit": "70%",
                    # Cache settings
                    "index.knn.space_type": vector_similarity,
                    # Preload caching
                    "index.store.preload": ["nvd", "dvd"]
                },
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "pbt_name": {
                            "type": "text", 
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"},
                                "ngram": {"type": "text", "analyzer": "ngram_analyzer"}
                            }
                        },
                        "pbt_definition": {
                            "type": "text", 
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "cdm": {
                            "type": "text", 
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": vector_dims,
                            "index": True,
                            "similarity": vector_similarity,
                            "index_options": {
                                "type": "hnsw",
                                "m": hnsw_m,
                                "ef_construction": hnsw_ef_construction
                            }
                        }
                    }
                }
            }
            
            # Create index with enhanced settings
            try:
                await client.indices.create(index=index_name, body=index_settings)
                logger.info(f"Index '{index_name}' created successfully with advanced vector search capabilities")
            except RequestError as e:
                if "resource_already_exists_exception" in str(e):
                    logger.warning(f"Index '{index_name}' already exists, skipping creation")
                else:
                    # Check if this is a compatibility error and try with simplified settings
                    logger.warning(f"Error creating index: {e}")
                    logger.info("Trying with simplified HNSW settings...")
                    
                    # Simplify settings based on Elasticsearch version
                    try:
                        # Remove knn-specific settings that might not be supported
                        if "index.knn" in index_settings["settings"]:
                            del index_settings["settings"]["index.knn"]
                        if "index.knn.memory_circuit_breaker.enabled" in index_settings["settings"]:
                            del index_settings["settings"]["index.knn.memory_circuit_breaker.enabled"]
                        if "index.knn.memory_circuit_breaker.limit" in index_settings["settings"]:
                            del index_settings["settings"]["index.knn.memory_circuit_breaker.limit"]
                        if "index.knn.space_type" in index_settings["settings"]:
                            del index_settings["settings"]["index.knn.space_type"]
                        
                        await client.indices.create(index=index_name, body=index_settings)
                        logger.info(f"Index '{index_name}' created successfully with simplified settings")
                    except Exception as e2:
                        logger.error(f"Failed to create index with simplified settings: {e2}")
                        raise
            
            # Create index alias
            alias_name = f"{index_name}_alias"
            logger.info(f"Creating alias '{alias_name}' for index '{index_name}'")
            try:
                await client.indices.put_alias(index=index_name, name=alias_name)
                logger.info(f"Alias '{alias_name}' created successfully")
            except Exception as e:
                logger.warning(f"Error creating alias: {e}")
            
        else:
            logger.info(f"Index '{index_name}' already exists")
            
            # Update index settings if applicable
            if not force_recreate:
                try:
                    # Get current settings
                    current_settings = await client.indices.get_settings(index=index_name)
                    logger.info(f"Current index settings retrieved")
                    
                    # Check for vector search capabilities
                    mapping = await client.indices.get_mapping(index=index_name)
                    properties = mapping.get(index_name, {}).get("mappings", {}).get("properties", {})
                    
                    # Check if embedding field exists and has vector search capabilities
                    if "embedding" in properties:
                        embedding_field = properties["embedding"]
                        if embedding_field.get("type") == "dense_vector" and embedding_field.get("index") is True:
                            logger.info("Vector search is enabled on the existing index")
                        else:
                            logger.warning("The embedding field exists but vector search is not configured correctly")
                            logger.warning("To enable vector search, recreate the index with the --force option")
                    else:
                        logger.warning("The embedding field does not exist in the current mapping")
                        logger.warning("To add the embedding field, recreate the index with the --force option")
                        
                except Exception as e:
                    logger.error(f"Error checking index configuration: {e}")
        
        logger.info("Elasticsearch initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Error initializing Elasticsearch: {e}")
        raise
    finally:
        # Close Elasticsearch client
        await client.close()
        logger.info("Elasticsearch client closed")

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Initialize Elasticsearch with advanced vector search capabilities")
    parser.add_argument("--force", action="store_true", help="Force recreate the Elasticsearch index")
    parser.add_argument("--vector-dims", type=int, default=3072, help="Vector dimensions for embedding field")
    parser.add_argument("--vector-similarity", choices=["cosine", "dot_product", "l2_norm"], default="cosine", 
                        help="Similarity function for vector search")
    parser.add_argument("--hnsw-m", type=int, default=16, help="Number of bi-directional links in HNSW graph")
    parser.add_argument("--hnsw-ef-construction", type=int, default=100, help="Controls quality of the graph during construction")
    parser.add_argument("--shards", type=int, default=1, help="Number of primary shards")
    parser.add_argument("--replicas", type=int, default=0, help="Number of replicas")
    args = parser.parse_args()
    
    # Load environment variables
    env = get_os_env()
    
    # Run the async function
    asyncio.run(init_elasticsearch(
        args.force, 
        args.vector_dims, 
        args.vector_similarity,
        args.hnsw_m, 
        args.hnsw_ef_construction, 
        args.shards, 
        args.replicas
    ))

if __name__ == "__main__":
    main()