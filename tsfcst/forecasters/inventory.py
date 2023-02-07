FORECASTERS = {
    "ma_12": {
        "model_name": "ma",
        "params_model": {
            "window": 12
        },
        "freq_model": "MS",
    },
    "ema_12": {
        "model_name": "ma",
        "params_model": {
            "average": "exponential",
            "window": 12
        },
        "freq_model": "MS",
    },
    "hw": {
        "model_name": "hw_sm",
        "params_model": {
            "seasonal_periods": 12,
        },
        "freq_model": "MS",
    },
    "hw_flex_trend_no_seas": {
        "model_name": "hw_sm",
        "params_model": {
            "seasonal": None,
            "smoothing_trend_max": 0.25,
            "smoothing_level_max": 0.25,
        },
        "freq_model": "MS",
    },
    "hw_flex_seas": {
        "model_name": "hw_sm",
        "params_model": {
            "seasonal_periods": 12,
            "smoothing_seasonal_max": 0.20,
        },
        "freq_model": "MS",
    },
    "hw_boxcox_0_10": {
        "model_name": "hw_sm",
        "freq_model": "MS",
        "boxcox_lambda": 0.10,
    },
    "hw_boxcox_0_50": {
        "model_name": "hw_sm",
        "params_model": {
            "seasonal_periods": 12,
            "smoothing_trend": (0.00, 0.03),
            "smoothing_level": (0.00, 0.03),
            "smoothing_seasonal": (0.00, 0.05),
        },
        "freq_model": "MS",
        "boxcox_lambda": 0.50,
    },
    "prophet": {
        "model_name": "prophet",
        "freq_model": "MS",
    },
}

DEFAULT_FALLBACK = "ema_12"

FORECASTERS_NAMES = list(FORECASTERS.keys())
