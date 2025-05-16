"""
Custom tiktoken initialization with local tokenizer file.
Save this file as app/utils/tiktoken_local.py
"""

import os
import sys
import logging
import json
import regex as re
from typing import Dict, List, Optional, Union, cast
import tiktoken
from tiktoken.load import load_tiktoken_bpe

logger = logging.getLogger(__name__)

# Path to the local tokenizer file
LOCAL_TOKENIZER_PATH = os.path.join("data", "cl100k_base.tiktoken")

# Original tiktoken registry - we'll modify this
original_registry = tiktoken._registry.DEFAULT_ENCODING_REGISTRY.copy()

def override_tiktoken_download():
    """Override tiktoken's download functionality to use local files."""
    try:
        # Function to load BPE from local file instead of downloading
        def custom_load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict:
            """Load a tiktoken BPE file from the local directory."""
            logger.info(f"Loading tokenizer from local file: {LOCAL_TOKENIZER_PATH}")
            
            # Check if the file exists
            if not os.path.exists(LOCAL_TOKENIZER_PATH):
                logger.error(f"Local tokenizer file not found: {LOCAL_TOKENIZER_PATH}")
                raise FileNotFoundError(f"Local tokenizer file not found: {LOCAL_TOKENIZER_PATH}")
            
            try:
                # Load the file content
                with open(LOCAL_TOKENIZER_PATH, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Parse the BPE data
                return {"model_path": LOCAL_TOKENIZER_PATH, "content": content}
            except Exception as e:
                logger.error(f"Error loading local tokenizer file: {e}")
                raise
        
        # Override the tiktoken load function
        tiktoken.load.load_tiktoken_bpe = custom_load_tiktoken_bpe
        
        # Test loading the model to ensure it works
        encoding = tiktoken.get_encoding("cl100k_base")
        sample_text = "Testing the tokenizer"
        tokens = encoding.encode(sample_text)
        logger.info(f"Successfully loaded local tokenizer. Sample tokenization: {sample_text} -> {tokens}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to override tiktoken download: {e}")
        return False

# Apply the override when this module is imported
override_applied = override_tiktoken_download()
if override_applied:
    logger.info("Tiktoken download override applied successfully.")
else:
    logger.warning("Failed to apply tiktoken download override.")