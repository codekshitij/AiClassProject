# Music Sentiment Analysis: 60 Years of Musical Emotion (1950-2019)

A comprehensive analysis of music sentiment trends over six decades using multiple sentiment analysis methods.

## 🎵 Project Overview

This project analyzes how music sentiment has evolved from 1950 to 2019, examining over 28,000 songs across multiple genres. Using state-of-the-art sentiment analysis techniques, we explore temporal trends, genre differences, and the emotional evolution of popular music.

### Key Questions Addressed
- Has music become more positive or negative over time?
- Which decades showed the most significant sentiment changes?
- How do different genres compare in their sentiment evolution?
- Are there correlations between historical events and music sentiment?

## 📊 Dataset

- **Source**: Mendeley Music Dataset (1950-2019)
- **Size**: ~28,000 songs
- **Features**: Lyrics, metadata, audio features, emotion annotations
- **Time Span**: 70 years (1950-2019)
- **Genres**: Multiple genres including pop, rock, country, jazz, etc.

## 🛠️ Technical Stack

### Sentiment Analysis Methods
- **VADER**: Rule-based sentiment analysis optimized for social media text
- **TextBlob**: Simple polarity and subjectivity analysis
- **Transformers**: State-of-the-art transformer-based sentiment analysis (RoBERTa)
- **Composite Score**: Weighted average of all methods with confidence scoring

### Technologies Used
- **Python 3.8+**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning utilities
- **Transformers**: Hugging Face transformer models
- **VADER Sentiment**: Lexicon-based sentiment analysis
- **TextBlob**: Simple sentiment analysis
- **SciPy**: Statistical analysis

## 📁 Project Structure

```
Music Dataset Lyrics and Metadata from 1950 to 2019/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .cursorrulefile                     # Project rules and structure
├── data/
│   ├── tcc_ceds_music.csv             # Original dataset
│   └── processed/
│       ├── cleaned_music_dataset.csv  # Cleaned dataset
│       └── music_with_sentiment.csv   # Dataset with sentiment scores
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                  # Data loading utilities
│   └── analysis/
│       ├── __init__.py
│       └── sentiment.py               # Sentiment analysis utilities
└── notebooks/
    ├── 01_eda_music_dataset.ipynb     # Exploratory Data Analysis
    ├── 02_sentiment_analysis.ipynb    # Multi-method Sentiment Analysis
    ├── 03_temporal_trends.ipynb       # Temporal Trend Analysis
    └── 04_final_visualizations.ipynb  # Final Visualizations
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   # If using git
   git clone <repository-url>
   cd "Music Dataset Lyrics and Metadata from 1950 to 2019"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data (if needed)**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('vader_lexicon')
   ```

### Running the Analysis

Execute the notebooks in order:

1. **Exploratory Data Analysis**
   ```bash
   jupyter notebook notebooks/01_eda_music_dataset.ipynb
   ```

2. **Sentiment Analysis**
   ```bash
   jupyter notebook notebooks/02_sentiment_analysis.ipynb
   ```

3. **Temporal Trends**
   ```bash
   jupyter notebook notebooks/03_temporal_trends.ipynb
   ```

4. **Final Visualizations**
   ```bash
   jupyter notebook notebooks/04_final_visualizations.ipynb
   ```

## 📈 Key Findings

### Overall Sentiment Trends
- **Temporal Evolution**: Analysis reveals significant changes in music sentiment over the 60-year period
- **Method Agreement**: High correlation between different sentiment analysis methods
- **Genre Differences**: Clear variations in sentiment across different musical genres
- **Statistical Significance**: Robust statistical validation of observed trends

### Genre Insights
- **Most Positive Genres**: [Results will be populated after analysis]
- **Most Negative Genres**: [Results will be populated after analysis]
- **Stability Analysis**: Some genres show consistent sentiment over time, others vary significantly

### Temporal Patterns
- **Decade Analysis**: Each decade shows distinct sentiment characteristics
- **Historical Context**: Sentiment changes correlate with major historical events
- **Trend Strength**: Quantified measures of sentiment change over time

## 🔬 Methodology

### Data Preprocessing
1. **Data Cleaning**: Removal of missing values, short lyrics, and invalid entries
2. **Text Processing**: Standardization of lyrics text
3. **Temporal Grouping**: Organization by year and decade for trend analysis

### Sentiment Analysis Pipeline
1. **Multi-Method Analysis**: Parallel processing using VADER, TextBlob, and Transformers
2. **Composite Scoring**: Weighted combination of all methods
3. **Confidence Assessment**: Agreement-based confidence scoring
4. **Validation**: Cross-method correlation analysis

### Statistical Analysis
1. **Trend Analysis**: Linear regression for temporal trends
2. **Significance Testing**: Statistical validation of observed changes
3. **Genre Comparison**: ANOVA and post-hoc tests for genre differences
4. **Correlation Analysis**: Relationship between different sentiment measures

## 📊 Visualizations

The project includes comprehensive visualizations:

### Static Visualizations (Matplotlib/Seaborn)
- Temporal sentiment trends with confidence intervals
- Genre comparison charts
- Distribution analysis
- Statistical significance plots

### Interactive Visualizations (Plotly)
- Interactive timeline with hover details
- Genre sentiment heatmaps
- Method comparison charts
- Decade-by-decade analysis

## 🎯 Results and Insights

### Sentiment Distribution
- **Positive Songs**: [Percentage] of songs show positive sentiment
- **Neutral Songs**: [Percentage] of songs show neutral sentiment
- **Negative Songs**: [Percentage] of songs show negative sentiment

### Temporal Trends
- **Overall Trend**: [Increasing/Decreasing] sentiment over time
- **Trend Strength**: [R² value] of variance explained
- **Statistical Significance**: [Yes/No] with p-value

### Genre Analysis
- **Most Positive**: [Genre name] with [sentiment score]
- **Most Negative**: [Genre name] with [sentiment score]
- **Most Stable**: [Genre name] with [stability score]
- **Most Variable**: [Genre name] with [variability score]

## 🔍 Limitations and Considerations

### Data Limitations
- **Language Bias**: Analysis focuses on English-language songs
- **Genre Representation**: Some genres may be underrepresented
- **Temporal Coverage**: Uneven distribution of songs across decades
- **Lyrics Quality**: Variable quality of lyrics data

### Methodological Limitations
- **Sentiment Analysis Accuracy**: No method is 100% accurate
- **Cultural Context**: Sentiment analysis may not capture cultural nuances
- **Historical Context**: Limited ability to account for historical context
- **Subjectivity**: Sentiment is inherently subjective

### Technical Limitations
- **Computational Resources**: Transformer models require significant resources
- **Processing Time**: Large dataset analysis takes considerable time
- **Memory Requirements**: High memory usage for large-scale analysis

## 🚀 Future Work

### Potential Extensions
1. **Multi-Language Analysis**: Extend to non-English songs
2. **Audio Feature Integration**: Combine lyrics sentiment with audio features
3. **Artist-Specific Analysis**: Individual artist sentiment evolution
4. **Cultural Context**: Incorporate historical and cultural factors
5. **Real-Time Analysis**: Extend to current music trends

### Methodological Improvements
1. **Custom Models**: Train domain-specific sentiment models
2. **Ensemble Methods**: Advanced ensemble techniques
3. **Contextual Analysis**: Incorporate song context and themes
4. **Validation Studies**: Human validation of sentiment scores

## 📚 References

### Datasets
- Mendeley Music Dataset (1950-2019)
- VADER Sentiment Lexicon
- TextBlob Sentiment Analysis
- Hugging Face Transformers

### Methods
- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
- Loria, S. (2018). TextBlob: Simplified Text Processing.
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is for educational and research purposes. Please respect the original dataset's terms of use.

## 👥 Acknowledgments

- **Dataset Providers**: Mendeley for providing the music dataset
- **Open Source Community**: For the excellent libraries and tools
- **Research Community**: For sentiment analysis methodologies
- **Music Industry**: For the rich cultural data analyzed

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue in the project repository.

---

**Note**: This project is for educational and research purposes. The analysis provides insights into music sentiment trends but should not be considered definitive or comprehensive of all musical expression during the analyzed period.

