"""
Data loading utilities for the music sentiment analysis project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

def load_music_dataset(data_path="data/tcc_ceds_music.csv"):
    """
    Load the music dataset with proper encoding and data types.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded and cleaned dataset
    """
    try:
        # Load the dataset with explicit encoding
        df = pd.read_csv(data_path, encoding='utf-8')
        
        # Basic data cleaning
        df = clean_dataset(df)
        
        print(f"Dataset loaded successfully: {df.shape[0]} songs, {df.shape[1]} features")
        return df
        
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        warnings.warn("UTF-8 encoding failed, trying latin-1")
        df = pd.read_csv(data_path, encoding='latin-1')
        df = clean_dataset(df)
        return df

def clean_dataset(df):
    """
    Clean the dataset by handling missing values and data types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset
    """
    # Remove rows with missing lyrics
    initial_count = len(df)
    df = df.dropna(subset=['lyrics'])
    
    # Remove rows with very short lyrics (less than 10 words)
    df = df[df['lyrics'].str.split().str.len() >= 10]
    
    # Convert release_date to datetime (handle as year if it's already numeric)
    if df['release_date'].dtype in ['int64', 'int32', 'float64']:
        # If release_date is already a year, use it directly
        df['year'] = df['release_date'].astype(int)
        # Create a proper datetime column for consistency
        df['release_date'] = pd.to_datetime(df['year'].astype(str) + '-01-01', errors='coerce')
    else:
        # If it's a string, try to parse as datetime
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        # Extract year for easier analysis
        df['year'] = df['release_date'].dt.year
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['release_date'])
    
    # Filter to 1950-2019 range as specified
    df = df[(df['year'] >= 1950) & (df['year'] <= 2019)]
    
    # Create decade column for analysis
    df['decade'] = (df['year'] // 10) * 10
    
    # Clean genre column
    df['genre'] = df['genre'].str.strip().str.lower()
    
    # Remove rows with empty genre
    df = df[df['genre'].notna() & (df['genre'] != '')]
    
    final_count = len(df)
    removed_count = initial_count - final_count
    
    print(f"Data cleaning completed:")
    print(f"  - Removed {removed_count} rows ({removed_count/initial_count*100:.1f}%)")
    print(f"  - Final dataset: {final_count} songs")
    print(f"  - Year range: {df['year'].min()}-{df['year'].max()}")
    print(f"  - Genres: {df['genre'].nunique()} unique genres")
    
    return df

def get_dataset_info(df):
    """
    Get basic information about the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
        
    Returns:
    --------
    dict
        Dataset information
    """
    info = {
        'total_songs': len(df),
        'year_range': (df['year'].min(), df['year'].max()),
        'genres': df['genre'].value_counts().to_dict(),
        'artists': df['artist_name'].nunique(),
        'avg_lyrics_length': df['lyrics'].str.split().str.len().mean(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    return info

def filter_by_criteria(df, min_year=None, max_year=None, genres=None, min_lyrics_length=None):
    """
    Filter dataset by various criteria.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    min_year : int, optional
        Minimum year
    max_year : int, optional
        Maximum year
    genres : list, optional
        List of genres to include
    min_lyrics_length : int, optional
        Minimum lyrics length in words
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataset
    """
    filtered_df = df.copy()
    
    if min_year is not None:
        filtered_df = filtered_df[filtered_df['year'] >= min_year]
    
    if max_year is not None:
        filtered_df = filtered_df[filtered_df['year'] <= max_year]
    
    if genres is not None:
        filtered_df = filtered_df[filtered_df['genre'].isin(genres)]
    
    if min_lyrics_length is not None:
        filtered_df = filtered_df[filtered_df['lyrics'].str.split().str.len() >= min_lyrics_length]
    
    return filtered_df

