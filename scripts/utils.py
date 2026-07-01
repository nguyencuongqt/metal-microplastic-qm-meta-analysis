"""
Utility Functions for Qm Meta-Analysis Project
==============================================

This module contains shared utility functions used across the analysis pipeline,
including configuration loading, logging setup, and common data operations.

Author: Manuscript authors
Date: March 2026
"""

import os
import sys
import logging
import yaml
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class ProjectConfig:
    """Load and manage project configuration."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Parameters
        ----------
        config_path : str
            Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.root_dir = Path(__file__).parent.parent
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Parameters
        ----------
        key : str
            Configuration key (e.g., 'modeling.bayesian.chains')
        default : Any
            Default value if key not found
            
        Returns
        -------
        Any
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def get_path(self, path_key: str) -> Path:
        """
        Get absolute path from configuration.
        
        Parameters
        ----------
        path_key : str
            Key in paths section of config
            
        Returns
        -------
        Path
            Absolute path
        """
        rel_path = self.config['paths'].get(path_key, '')
        return self.root_dir / rel_path
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self.get(key)


def setup_logging(config: ProjectConfig, 
                  script_name: str,
                  log_to_file: bool = True,
                  log_to_console: bool = True) -> logging.Logger:
    """
    Set up logging for a script.
    
    Parameters
    ----------
    config : ProjectConfig
        Project configuration object
    script_name : str
        Name of the calling script
    log_to_file : bool
        Whether to log to file
    log_to_console : bool
        Whether to log to console
        
    Returns
    -------
    logging.Logger
        Configured logger object
    """
    logger = logging.getLogger(script_name)
    logger.setLevel(getattr(logging, config.get('logging.level', 'INFO')))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        config.get('logging.format', 
                  '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # File handler
    if log_to_file:
        log_dir = config.get_path('logs')
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{script_name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def set_random_seed(seed: Optional[int] = None, config: Optional[ProjectConfig] = None) -> None:
    """
    Set random seeds for reproducibility.
    
    Parameters
    ----------
    seed : int, optional
        Random seed value
    config : ProjectConfig, optional
        Configuration object to get seed from if not provided
    """
    if seed is None and config is not None:
        seed = config.get('random_seed', 42)
    elif seed is None:
        seed = 42
    
    np.random.seed(seed)
    
    # Try to set seeds for other libraries if available
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    # Note: pytensor.config.optimizer must be set BEFORE importing pymc
    # Scripts that use PyMC should configure pytensor at the top of the file


def save_dataframe(df: pd.DataFrame, 
                   filepath: Union[str, Path],
                   **kwargs) -> None:
    """
    Save DataFrame with automatic format detection.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str or Path
        Output file path
    **kwargs
        Additional arguments passed to save function
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    extension = filepath.suffix.lower()
    if extension == '.csv':
        df.to_csv(filepath, **kwargs)
    elif extension in ['.xlsx', '.xls']:
        df.to_excel(filepath, **kwargs)
    elif extension == '.parquet':
        df.to_parquet(filepath, **kwargs)
    elif extension == '.feather':
        df.to_feather(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {extension}")


def load_dataframe(filepath: Union[str, Path],
                   **kwargs) -> pd.DataFrame:
    """
    Load DataFrame with automatic format detection.
    
    Parameters
    ----------
    filepath : str or Path
        Input file path
    **kwargs
        Additional arguments passed to load function
        
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    extension = filepath.suffix.lower()
    if extension == '.csv':
        return pd.read_csv(filepath, **kwargs)
    elif extension in ['.xlsx', '.xls']:
        return pd.read_excel(filepath, **kwargs)
    elif extension == '.parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif extension == '.feather':
        return pd.read_feather(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {extension}")


def save_model(model: Any, 
               filepath: Union[str, Path],
               metadata: Optional[Dict] = None) -> None:
    """
    Save model object with metadata.
    
    Parameters
    ----------
    model : Any
        Model object to save
    filepath : str or Path
        Output file path
    metadata : dict, optional
        Additional metadata to save with model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model': model,
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat()
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f)


def load_model(filepath: Union[str, Path]) -> tuple:
    """
    Load model object with metadata.
    
    Parameters
    ----------
    filepath : str or Path
        Input file path
        
    Returns
    -------
    tuple
        (model, metadata) tuple
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        save_dict = pickle.load(f)
    
    return save_dict['model'], save_dict.get('metadata', {})


def save_json(data: Dict, 
              filepath: Union[str, Path],
              indent: int = 2) -> None:
    """
    Save dictionary as JSON.
    
    Parameters
    ----------
    data : dict
        Data to save
    filepath : str or Path
        Output file path
    indent : int
        JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, np.generic):
            # Fallback for other numpy types (np.complex_, etc.)
            return obj.item()
        return obj
    
    data = convert_types(data)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Load JSON file.
    
    Parameters
    ----------
    filepath : str or Path
        Input file path
        
    Returns
    -------
    dict
        Loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def get_timestamp(format: str = '%Y%m%d_%H%M%S') -> str:
    """
    Get current timestamp as string.
    
    Parameters
    ----------
    format : str
        Datetime format string
        
    Returns
    -------
    str
        Formatted timestamp
    """
    return datetime.now().strftime(format)


def create_output_path(config: ProjectConfig,
                       output_type: str,
                       filename: str,
                       timestamp: bool = True) -> Path:
    """
    Create standardized output path.
    
    Parameters
    ----------
    config : ProjectConfig
        Project configuration
    output_type : str
        Type of output ('models', 'results', 'figures', etc.)
    filename : str
        Base filename
    timestamp : bool
        Whether to add timestamp to filename
        
    Returns
    -------
    Path
        Full output path
    """
    base_dir = config.get_path(output_type)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if timestamp and config.get('output.timestamp_outputs', True):
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        filename = f"{stem}_{get_timestamp()}{suffix}"
    
    return base_dir / filename


def print_section_header(title: str, 
                        width: int = 80,
                        logger: Optional[logging.Logger] = None) -> None:
    """
    Print formatted section header.
    
    Parameters
    ----------
    title : str
        Section title
    width : int
        Total width of header
    logger : logging.Logger, optional
        Logger to use instead of print
    """
    header = f"\n{'=' * width}\n{title.center(width)}\n{'=' * width}\n"
    if logger:
        logger.info(header)
    else:
        print(header)


def check_required_columns(df: pd.DataFrame, 
                          required: List[str],
                          logger: Optional[logging.Logger] = None) -> bool:
    """
    Check if DataFrame has required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check
    required : list
        List of required column names
    logger : logging.Logger, optional
        Logger for warnings
        
    Returns
    -------
    bool
        True if all columns present
    """
    missing = set(required) - set(df.columns)
    if missing:
        message = f"Missing required columns: {missing}"
        if logger:
            logger.error(message)
        else:
            print(f"ERROR: {message}")
        return False
    return True


def describe_dataframe(df: pd.DataFrame, 
                       name: str = "DataFrame",
                       logger: Optional[logging.Logger] = None) -> None:
    """
    Log detailed DataFrame description.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to describe
    name : str
        Name for logging
    logger : logging.Logger, optional
        Logger to use
    """
    info = [
        f"\n{name} Summary:",
        f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        f"  Missing values: {df.isnull().sum().sum()} ({100*df.isnull().sum().sum()/df.size:.2f}%)",
    ]
    
    message = "\n".join(info)
    if logger:
        logger.info(message)
    else:
        print(message)


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities module...")
    
    # Test config loading
    try:
        config = ProjectConfig()
        print(f"[PASS] Configuration loaded successfully")
        print(f"  Random seed: {config.get('random_seed')}")
        print(f"  Version: {config.get('version')}")
    except Exception as e:
        print(f"[FAIL] Configuration loading failed: {e}")
    
    # Test logging
    try:
        logger = setup_logging(config, "test_utils")
        logger.info("Test log message")
        print("[PASS] Logging setup successful")
    except Exception as e:
        print(f"[FAIL] Logging setup failed: {e}")
    
    # Test random seed
    try:
        set_random_seed(config=config)
        print("[PASS] Random seed set successfully")
    except Exception as e:
        print(f"[FAIL] Random seed setting failed: {e}")
    
    print("\nAll utility tests completed.")
