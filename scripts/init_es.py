"""
Script to initialize Elasticsearch with proper index settings and mappings.
"""

import os
import sys
import asyncio
import argparse
import logging
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError
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

async def init_elasticsearch(force_recreate=False, analyzer_config=None, shards=1, replicas=0):
    """
    Initialize Elasticsearch with proper index settings and mappings.
    
    Args:
        force_recreate: Whether to force recreate the index if it exists
        analyzer_config: Custom analyzer configuration
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
    
    # Set up authentication
    auth = None
    if username and password:
        auth = (username, password)
    
    # Create Elasticsearch client
    client = AsyncElasticsearch(
        hosts=hosts,
        basic_auth=auth,
        verify_certs=False,
        request_timeout=60
    )
    
    try:
        logger.info(f"Connecting to Elasticsearch at {', '.join(hosts)}")
        
        # Check if Elasticsearch is running
        info = await client.info()
        es_version = info["version"]["number"]
        logger.info(f"Connected to Elasticsearch version {es_version}")
        
        # Check if index exists
        index_exists = await client.indices.exists(index=index_name)
        
        # Delete index if it exists and force_recreate is True
        if index_exists and force_recreate:
            logger.info(f"Deleting existing index '{index_name}'")
            await client.indices.delete(index=index_name)
            index_exists = False
        
        # Create index if it doesn't exist
        if not index_exists:
            logger.info(f"Creating index '{index_name}'")
            
            # Default analyzer configuration
            default_analyzer = {
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
            
            # Use custom analyzer if provided
            analyzer = analyzer_config if analyzer_config else default_analyzer
            
            # Index settings
            settings_config = {
                "settings": {
                    "number_of_shards": shards,
                    "number_of_replicas": replicas,
                    "analysis": analyzer,
                    "index.mapping.coerce": True,
                    "index.mapping.ignore_malformed": True
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
                            "analyzer": "standard"
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
                            "dims": 3072,  # Dimension size for text-embedding-3-large
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                }
            }
            
            # Create index with settings
            await client.indices.create(index=index_name, body=settings_config)
            logger.info(f"Index '{index_name}' created successfully with {shards} shards and {replicas} replicas")
            
            # Check if index was created successfully
            index_exists = await client.indices.exists(index=index_name)
            if index_exists:
                logger.info(f"Verified index '{index_name}' exists")
                
                # Get index settings and mappings
                index_settings = await client.indices.get_settings(index=index_name)
                logger.info(f"Index settings: {index_settings}")
                
                index_mappings = await client.indices.get_mapping(index=index_name)
                logger.info(f"Index mappings: {index_mappings}")
            else:
                logger.error(f"Failed to create index '{index_name}'")
        else:
            logger.info(f"Index '{index_name}' already exists")
            
            # Update settings if needed
            if not force_recreate:
                logger.info(f"Updating index settings and mappings not implemented. Use --force to recreate the index with new settings.")
        
        # Create index aliases if needed
        alias_name = f"{index_name}_alias"
        alias_exists = False
        
        try:
            alias_info = await client.indices.get_alias(name=alias_name)
            alias_exists = True
        except NotFoundError:
            alias_exists = False
        
        if not alias_exists:
            await client.indices.put_alias(index=index_name, name=alias_name)
            logger.info(f"Created alias '{alias_name}' for index '{index_name}'")
        else:
            logger.info(f"Alias '{alias_name}' already exists")
        
        # Create ILM policy (Index Lifecycle Management)
        policy_name = f"{index_name}_policy"
        policy_exists = False
        
        try:
            # Check if ILM policy exists
            policy_info = await client.ilm.get_lifecycle(name=policy_name)
            policy_exists = True
        except NotFoundError:
            policy_exists = False
        except Exception as e:
            # ILM might not be available in all Elasticsearch versions
            logger.warning(f"ILM might not be available: {e}")
        
        if not policy_exists:
            try:
                # Create ILM policy
                ilm_policy = {
                    "policy": {
                        "phases": {
                            "hot": {
                                "min_age": "0ms",
                                "actions": {
                                    "set_priority": {
                                        "priority": 100
                                    }
                                }
                            },
                            "warm": {
                                "min_age": "30d",
                                "actions": {
                                    "set_priority": {
                                        "priority": 50
                                    },
                                    "shrink": {
                                        "number_of_shards": 1
                                    },
                                    "forcemerge": {
                                        "max_num_segments": 1
                                    }
                                }
                            },
                            "cold": {
                                "min_age": "60d",
                                "actions": {
                                    "set_priority": {
                                        "priority": 0
                                    },
                                    "freeze": {}
                                }
                            },
                            "delete": {
                                "min_age": "90d",
                                "actions": {
                                    "delete": {}
                                }
                            }
                        }
                    }
                }
                
                await client.ilm.put_lifecycle(name=policy_name, body=ilm_policy)
                logger.info(f"Created ILM policy '{policy_name}'")
            except Exception as e:
                logger.warning(f"Could not create ILM policy: {e}")
        else:
            logger.info(f"ILM policy '{policy_name}' already exists")
        
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
    parser = argparse.ArgumentParser(description="Initialize Elasticsearch with proper index settings and mappings")
    parser.add_argument("--force", action="store_true", help="Force recreate the Elasticsearch index")
    parser.add_argument("--shards", type=int, default=1, help="Number of primary shards")
    parser.add_argument("--replicas", type=int, default=0, help="Number of replicas")
    args = parser.parse_args()
    
    # Load environment variables
    env = get_os_env()
    
    # Run the async function
    asyncio.run(init_elasticsearch(args.force, None, args.shards, args.replicas))

if __name__ == "__main__":
    main()