from src.utils.config import SPORTSDATA_API_KEY, DATA_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("test_config", "test_config.log")

logger.info(f"DATA_DIR: {DATA_DIR}")
if SPORTSDATA_API_KEY:
    logger.info("SPORTSDATA_API_KEY yüklendi (değerini log'a yazmıyorum).")
else:
    logger.error("SPORTSDATA_API_KEY BOŞ! .env dosyasını kontrol et.")
