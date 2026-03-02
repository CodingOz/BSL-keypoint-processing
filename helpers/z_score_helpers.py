import numpy as np


def zscore_normalize(data):
    """
    Z-score normalization: (x - mean) / std
    makes data dimensionless with mean=0, std=1
    """
    data_array = np.array(data)

    valid_data = data_array[~np.isnan(data_array)]
    
    if len(valid_data) == 0:
        return data_array
    
    mean = np.nanmean(valid_data)
    std = np.nanstd(valid_data)
    
    if std == 0:
        return data_array - mean  # Avoid division by zero
    
    return (data_array - mean) / std


def minmax_normalize(data):
    """
    Min-max normalization: (x - min) / (max - min)
    scales data to range [0, 1]
    """
    data_array = np.array(data)
    valid_data = data_array[~np.isnan(data_array)]
    
    if len(valid_data) == 0:
        return data_array
    
    min_val = np.nanmin(valid_data)
    max_val = np.nanmax(valid_data)
    
    if max_val == min_val:
        return np.zeros_like(data_array)
    
    return (data_array - min_val) / (max_val - min_val)


def peak_normalize(data):
    """
    Normalize by peak value.
    """
    data_array = np.array(data)
    valid_data = data_array[~np.isnan(data_array)]
    
    if len(valid_data) == 0:
        return data_array
    
    peak = np.nanmax(np.abs(valid_data))
    
    if peak == 0:
        return data_array
    
    return data_array / peak


def robust_zscore_normalize(data):
    """
    Robust z-score using Median Absolute Deviation.
    """
    data_array = np.array(data)
    valid_data = data_array[~np.isnan(data_array)]
    
    if len(valid_data) == 0:
        return data_array
    
    median = np.nanmedian(valid_data)
    mad = np.nanmedian(np.abs(valid_data - median))
    
    if mad == 0:
        return data_array - median
    
    # 1.4826 makes MAD comparable to standard deviation
    return (data_array - median) / (1.4826 * mad)