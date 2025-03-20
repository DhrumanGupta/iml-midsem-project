import sys
from loguru import logger

logger.remove()  # Remove default handler
logger.add(
    sink=sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)
