import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name, log_file=None):
    """
    Projede kullanılacak standart logger.
    - name: logger adı (örn. 'data_acquisition', 'elo_rating')
    - log_file: logs/ altında kullanılacak dosya adı. None ise sadece console log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Aynı logger'a tekrar handler eklemeyelim
    if logger.handlers:
        return logger

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # File handler (isteğe bağlı)
    if log_file:
        log_path = LOG_DIR / log_file
        fh = RotatingFileHandler(
            log_path,
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8"
        )
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger
