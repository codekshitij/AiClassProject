"""
Sentiment analysis utilities for music lyrics.
Supports multiple sentiment analysis methods: VADER, TextBlob, and Transformers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import sentiment analysis libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: VADER sentiment analysis not available. Install with: pip install vaderSentiment")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: TextBlob not available. Install with: pip install textblob")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install transformers torch")

class MusicSentimentAnalyzer:
    """
    A comprehensive sentiment analyzer for music lyrics using multiple methods.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the sentiment analyzer.
        
        Parameters:
        -----------
        use_gpu : bool
            Whether to use GPU for transformer models
        """
        self.use_gpu = use_gpu and torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False
        self.vader_analyzer = None
        self.transformer_pipeline = None
        
        # Initialize available analyzers
        self._initialize_analyzers()
    
    def _initialize_analyzers(self):
        """Initialize available sentiment analyzers."""
        # Initialize VADER
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
            print("✓ VADER sentiment analyzer initialized")
        
        # Initialize Transformers pipeline
        if TRANSFORMERS_AVAILABLE:
            try:
                device = 0 if self.use_gpu else -1
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=device,
                    return_all_scores=True
                )
                print("✓ Transformer sentiment analyzer initialized")
            except Exception as e:
                print(f"Warning: Could not initialize transformer model: {e}")
                self.transformer_pipeline = None
    
    def analyze_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            VADER sentiment scores
        """
        if not VADER_AVAILABLE or self.vader_analyzer is None:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
        
        scores = self.vader_analyzer.polarity_scores(text)
        return scores
    
    def analyze_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            TextBlob sentiment scores
        """
        if not TEXTBLOB_AVAILABLE:
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_transformer(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using transformer model.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Transformer sentiment scores
        """
        if not TRANSFORMERS_AVAILABLE or self.transformer_pipeline is None:
            return {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        
        try:
            # Truncate text if too long (transformer models have token limits)
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            results = self.transformer_pipeline(text)
            
            # Convert results to dictionary
            sentiment_scores = {}
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                sentiment_scores[label] = score
            
            return sentiment_scores
            
        except Exception as e:
            print(f"Error in transformer analysis: {e}")
            return {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
    
    def analyze_single_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text using all available methods.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Combined sentiment scores from all methods
        """
        if not text or pd.isna(text):
            return self._get_empty_scores()
        
        # Clean text
        text = str(text).strip()
        if len(text) < 3:
            return self._get_empty_scores()
        
        results = {}
        
        # VADER analysis
        vader_scores = self.analyze_vader(text)
        results.update({f'vader_{k}': v for k, v in vader_scores.items()})
        
        # TextBlob analysis
        textblob_scores = self.analyze_textblob(text)
        results.update({f'textblob_{k}': v for k, v in textblob_scores.items()})
        
        # Transformer analysis
        transformer_scores = self.analyze_transformer(text)
        results.update({f'transformer_{k}': v for k, v in transformer_scores.items()})
        
        # Calculate composite scores
        results.update(self._calculate_composite_scores(results))
        
        return results
    
    def _get_empty_scores(self) -> Dict[str, float]:
        """Get empty sentiment scores for invalid text."""
        return {
            'vader_compound': 0.0, 'vader_pos': 0.0, 'vader_neu': 0.0, 'vader_neg': 0.0,
            'textblob_polarity': 0.0, 'textblob_subjectivity': 0.0,
            'transformer_positive': 0.0, 'transformer_neutral': 0.0, 'transformer_negative': 0.0,
            'composite_sentiment': 0.0, 'composite_confidence': 0.0
        }
    
    def _calculate_composite_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate composite sentiment scores from individual method scores.
        
        Parameters:
        -----------
        scores : dict
            Individual sentiment scores
            
        Returns:
        --------
        dict
            Composite scores
        """
        composite = {}
        
        # Composite sentiment (weighted average of available methods)
        sentiment_values = []
        weights = []
        
        # VADER compound score
        if 'vader_compound' in scores:
            sentiment_values.append(scores['vader_compound'])
            weights.append(0.4)
        
        # TextBlob polarity
        if 'textblob_polarity' in scores:
            sentiment_values.append(scores['textblob_polarity'])
            weights.append(0.3)
        
        # Transformer (positive - negative)
        if 'transformer_positive' in scores and 'transformer_negative' in scores:
            transformer_sentiment = scores['transformer_positive'] - scores['transformer_negative']
            sentiment_values.append(transformer_sentiment)
            weights.append(0.3)
        
        if sentiment_values:
            composite['composite_sentiment'] = np.average(sentiment_values, weights=weights)
        else:
            composite['composite_sentiment'] = 0.0
        
        # Composite confidence (based on agreement between methods)
        if len(sentiment_values) > 1:
            std_dev = np.std(sentiment_values)
            composite['composite_confidence'] = max(0.0, 1.0 - std_dev)
        else:
            composite['composite_confidence'] = 0.5
        
        return composite
    
    def analyze_dataframe(self, df: pd.DataFrame, lyrics_column: str = 'lyrics', 
                         batch_size: int = 100, progress_callback=None) -> pd.DataFrame:
        """
        Analyze sentiment for all lyrics in a DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing lyrics
        lyrics_column : str
            Name of the column containing lyrics
        batch_size : int
            Batch size for processing
        progress_callback : callable, optional
            Callback function for progress updates
            
        Returns:
        --------
        pd.DataFrame
            Original DataFrame with added sentiment columns
        """
        if lyrics_column not in df.columns:
            raise ValueError(f"Column '{lyrics_column}' not found in DataFrame")
        
        print(f"Analyzing sentiment for {len(df)} songs...")
        
        # Initialize result columns
        result_df = df.copy()
        sentiment_columns = [
            'vader_compound', 'vader_pos', 'vader_neu', 'vader_neg',
            'textblob_polarity', 'textblob_subjectivity',
            'transformer_positive', 'transformer_neutral', 'transformer_negative',
            'composite_sentiment', 'composite_confidence'
        ]
        
        for col in sentiment_columns:
            result_df[col] = 0.0
        
        # Process in batches
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            
            batch_df = df.iloc[start_idx:end_idx]
            
            for idx, row in batch_df.iterrows():
                lyrics = row[lyrics_column]
                sentiment_scores = self.analyze_single_text(lyrics)
                
                for col, value in sentiment_scores.items():
                    result_df.at[idx, col] = value
            
            # Progress update
            if progress_callback:
                progress_callback(batch_idx + 1, total_batches)
            else:
                print(f"Processed batch {batch_idx + 1}/{total_batches} ({end_idx}/{len(df)} songs)")
        
        print("Sentiment analysis completed!")
        return result_df
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for sentiment scores in a DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with sentiment columns
            
        Returns:
        --------
        dict
            Summary statistics for each sentiment method
        """
        sentiment_columns = [col for col in df.columns if any(method in col for method in ['vader', 'textblob', 'transformer', 'composite'])]
        
        summary = {}
        for col in sentiment_columns:
            if col in df.columns:
                summary[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                }
        
        return summary

def analyze_music_sentiment(df: pd.DataFrame, lyrics_column: str = 'lyrics', 
                          use_gpu: bool = False, batch_size: int = 100) -> pd.DataFrame:
    """
    Convenience function to analyze sentiment for music lyrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing music data
    lyrics_column : str
        Name of the column containing lyrics
    use_gpu : bool
        Whether to use GPU for transformer models
    batch_size : int
        Batch size for processing
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added sentiment analysis columns
    """
    analyzer = MusicSentimentAnalyzer(use_gpu=use_gpu)
    return analyzer.analyze_dataframe(df, lyrics_column, batch_size)

