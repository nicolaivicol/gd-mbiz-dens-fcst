# ******************************************************************************
# This contains all configs/parameters used in this project.
# ******************************************************************************

from pathlib import Path
import os
import logging

VERSION = '0.1.0'

# Directories
# ******************************************************************************
DIR_PROJ = (Path(__file__) / '..').resolve()
DIR_DATA = f'{DIR_PROJ}/data'
DIR_ARTIFACTS = f'{DIR_PROJ}/artifacts'
os.makedirs(DIR_ARTIFACTS, exist_ok=True)

# Logging
# ******************************************************************************
LOGS_LEVEL = logging.DEBUG
FILE_LOGS = f'{DIR_ARTIFACTS}/logs.log'
# set logging config:
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(FILE_LOGS), logging.StreamHandler()],
    level=LOGS_LEVEL,
)

HEIGHT_PLOT_LOW = 300
HEIGHT_PLOT_MEDIUM = 450
HEIGHT_PLOT_HIGH = 750
HEIGHT_PLOT_VERY_HIGH = 1200

PLOT_MARGINS_SMALL = dict(l=25, r=25, t=25, b=10)
PLOT_MARGINS_MEDIUM = dict(l=30, r=30, t=30, b=25)

MAKE_PLOTS = False

DIR_CACHE_TUNE_HYPER_PARAMS_W_OPTUNA = f'{DIR_ARTIFACTS}/cache/paramsfinder-findbest'

CV_ARGS_DICT = {
    'test': {
        'n_train_dates': 3,
        'step_train_dates': 3,
        'periods_val': 5,
        'periods_test': 0,
        'periods_out': 0,
        'periods_val_last': 3,
        'offset_last_date': 0
    },
    # 'cv_test': {
    #     'n_train_dates': 3,
    #     'step_train_dates': 3,
    #     'periods_val': 5,
    #     'periods_test': 5,
    #     'periods_out': 0,
    #     'periods_val_last': 3,
    #     'offset_last_date': 0
    # },
    'full': {
        'n_train_dates': 3,
        'step_train_dates': 3,
        'periods_val': 5,
        'periods_test': 0,
        'periods_out': 0,
        'periods_val_last': 5,
        'offset_last_date': 0
    }
}

SEARCH_ARGS_DICT = {
    'trend_level_damp': {
        'trend': True,
        'seasonal': False,
        'multiplicative': False,
        'level': True,
        'damp': True
    },
    'level': {
        'trend': False,
        'seasonal': False,
        'multiplicative': False,
        'level': True,
        'damp': False
    }
}
