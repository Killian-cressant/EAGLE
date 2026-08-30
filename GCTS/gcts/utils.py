import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
#import torch
from colorama import Fore



def get_logger(logger_name: str | None = None):
    if logger_name is None:
        logger_name = os.path.basename(__file__).split(".")[0]
    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(name)-5s %(levelname)-8s %(message)s")
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    return logger