import sys
from loguru import logger

logger.remove()

logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> | <white>{message}</white>",
    level="INFO"
)

logger.add(
    "logs/arxivlens.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days"
)