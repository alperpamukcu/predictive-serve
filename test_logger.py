from src.utils.logging_utils import get_logger

logger = get_logger("test_logger", "test_logger.log")

logger.info("Logger düzgün çalışıyor mu?")
logger.warning("Bu bir uyarı mesajıdır.")
logger.error("Bu bir hata mesajıdır.")
