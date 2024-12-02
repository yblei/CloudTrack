import logging
from loguru import logger
import sys
import hydra

# forward loguru logging to python logging
class PropagateHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)
    

# forward uncought exceptions to python logging
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    #sys.__excepthook__(exc_type, exc_value, exc_traceback)

#sys.excepthook = handle_exception

def fix_logging():
    # remove preexisting logger to avoid duplicate console messages
    logger.remove()

    logging.getLogger().removeHandler(logging.getLogger().handlers[0])
    logger.add(PropagateHandler(), format="{message}")
    logger.add(sys.stderr, level="ERROR")




