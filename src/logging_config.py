"""
Logging configuration - Clean and focused
"""

import logging
import sys

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Suppress noisy library logs
for lib in ["httpcore", "httpx", "urllib3", "chromadb"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

# App logger
logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)
