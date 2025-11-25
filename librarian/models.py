import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalysis:
    """
    Advanced statistical analysis for movie data
    """
    
    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize
        
        Args:
            movies_df: DataFrame containing processed movie data
        """
        self.movies = movies_df.copy()
    
    def correlation_analysis(self, variables: list = None) -> pd.DataFrame:
        """
        Analyze correlation between variables
        
        Args:
            variables: List of variables to analyze
            
        Returns:
            Correlation matrix
        """
        if variables is None:
            variables = ['budget', 'revenue', 'runtime', 'vote_average', 'popularity']
        
        numeric_cols = self.movies[variables].select_dtypes(include=[np.number])
        numeric_cols = numeric_cols.dropna()
        
        return numeric_cols.corr()
    
    def statistical_summary(self, column: str) -> dict:
        """
        Get detailed statistics about a column
        
        Args:
            column: Column name
            
        Returns:
            Dictionary containing statistics
        """
        data = pd.to_numeric(self.movies[column], errors='coerce').dropna()
        
        return {
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75),
            'max': data.max(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis()
        }
    
    def outlier_detection(self, column: str, method: str = 'iqr') -> np.ndarray:
        """
        Detect outlier values
        
        Args:
            column: Column name
            method: 'iqr' or 'zscore'
            
        Returns:
            Boolean array indicating outliers
        """
        data = pd.to_numeric(self.movies[column], errors='coerce')
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            return (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
        
        elif method == 'zscore':
            return np.abs(stats.zscore(data.dropna())) > 3
    
    def hypothesis_test_ttest(self, group1: str, group2: str, variable: str) -> dict:
        """
        Perform t-test between two groups
        
        Args:
            group1, group2: Group names
            variable: Variable to compare
            
        Returns:
            Dictionary containing test results
        """
        data1 = pd.to_numeric(self.movies[self.movies[group1]][variable], errors='coerce').dropna()
        data2 = pd.to_numeric(self.movies[~self.movies[group1]][variable], errors='coerce').dropna()
        
        t_stat, p_value = stats.ttest_ind(data1, data2)
        
        return {
            'group1_mean': data1.mean(),
            'group2_mean': data2.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def regression_coefficients(self, target: str, features: list) -> dict:
        """
        Calculate linear regression coefficients
        
        Args:
            target: Dependent variable
            features: List of independent variables
            
        Returns:
            Dictionary containing coefficients
        """
        from sklearn.linear_model import LinearRegression
        
        X = self.movies[features].select_dtypes(include=[np.number])
        y = pd.to_numeric(self.movies[target], errors='coerce')
        
        # Remove NaN
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        model = LinearRegression()
        model.fit(X, y)
        
        return {
            'coefficients': dict(zip(features, model.coef_)),
            'intercept': model.intercept_,
            'r_squared': model.score(X, y)
        }


class SegmentationAnalysis:
    """
    Movie segmentation analysis
    """
    
    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize
        
        Args:
            movies_df: DataFrame containing processed movie data
        """
        self.movies = movies_df.copy()
    
    def segment_by_budget(self, bins: int = 4) -> pd.DataFrame:
        """
        Segment movies by budget
        
        Args:
            bins: Number of segments
            
        Returns:
            DataFrame with segments
        """
        budget_col = pd.to_numeric(self.movies['budget'], errors='coerce')
        self.movies['budget_segment'] = pd.qcut(
            budget_col, 
            q=bins, 
            labels=['Low', 'Medium', 'High', 'Very High'][:bins],
            duplicates='drop'
        )
        
        segment_analysis = self.movies.groupby('budget_segment', observed=True).agg({
            'revenue': ['mean', 'count'],
            'vote_average': 'mean',
            'profit': 'mean'
        }).round(2)
        
        return segment_analysis
    
    def segment_by_rating(self, bins: int = 3) -> pd.DataFrame:
        """
        Segment movies by rating
        
        Args:
            bins: Number of segments
            
        Returns:
            DataFrame with segments
        """
        rating_col = pd.to_numeric(self.movies['vote_average'], errors='coerce')
        self.movies['rating_segment'] = pd.qcut(
            rating_col, 
            q=bins, 
            labels=['Low', 'Medium', 'High'][:bins],
            duplicates='drop'
        )
        
        segment_analysis = self.movies.groupby('rating_segment', observed=True).agg({
            'revenue': 'mean',
            'budget': 'mean',
            'id': 'count'
        }).round(2)
        
        return segment_analysis
    
    def segment_by_runtime(self, bins: int = 4) -> pd.DataFrame:
        """
        Segment movies by runtime
        
        Args:
            bins: Number of segments
            
        Returns:
            DataFrame with segments
        """
        runtime_col = pd.to_numeric(self.movies['runtime'], errors='coerce')
        self.movies['runtime_segment'] = pd.qcut(
            runtime_col, 
            q=bins, 
            labels=['Short', 'Normal', 'Long', 'Very Long'][:bins],
            duplicates='drop'
        )
        
        segment_analysis = self.movies.groupby('runtime_segment', observed=True).agg({
            'revenue': 'mean',
            'vote_average': 'mean',
            'budget': 'mean',
            'id': 'count'
        }).round(2)
        
        return segment_analysis
    
    def cross_segment_analysis(self, seg1: str, seg2: str, metric: str = 'revenue') -> pd.DataFrame:
        """
        Cross-analyze two segments
        
        Args:
            seg1, seg2: Segment names
            metric: Metric to analyze
            
        Returns:
            Pivot table containing results
        """
        cross_tab = pd.crosstab(
            self.movies[seg1], 
            self.movies[seg2], 
            values=pd.to_numeric(self.movies[metric], errors='coerce'),
            aggfunc='mean'
        ).round(2)
        
        return cross_tab


class TrendAnalysis:
    """
    Time-based trend analysis
    """
    
    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize
        
        Args:
            movies_df: DataFrame containing processed movie data
        """
        self.movies = movies_df.copy()
        self.movies['year'] = pd.to_datetime(self.movies['release_date'], errors='coerce').dt.year
    
    def moving_average(self, column: str, window: int = 5) -> pd.DataFrame:
        """
        Calculate moving average
        
        Args:
            column: Column name
            window: Window size
            
        Returns:
            DataFrame containing moving average
        """
        yearly_data = self.movies.groupby('year', observed=True).agg({
            column: 'mean'
        }).reset_index()
        
        yearly_data['moving_avg'] = yearly_data[column].rolling(
            window=window, 
            center=True
        ).mean()
        
        return yearly_data
    
    def year_over_year_growth(self, column: str) -> pd.DataFrame:
        """
        Calculate year-over-year growth
        
        Args:
            column: Column name
            
        Returns:
            DataFrame containing YoY growth
        """
        yearly_data = self.movies.groupby('year', observed=True).agg({
            column: 'mean'
        }).reset_index()
        
        yearly_data['yoy_growth'] = yearly_data[column].pct_change() * 100
        
        return yearly_data
    
    def seasonal_trend(self, column: str) -> pd.DataFrame:
        """
        Analyze seasonal trends
        
        Args:
            column: Column name
            
        Returns:
            DataFrame containing seasonal trends
        """
        month = pd.to_datetime(self.movies['release_date'], errors='coerce').dt.month
        
        seasonal_data = self.movies.groupby(month).agg({
            column: ['mean', 'median', 'std']
        }).round(2)
        
        return seasonal_data


class PerformanceMetrics:
    """
    Calculate performance metrics
    """
    
    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize
        
        Args:
            movies_df: DataFrame containing processed movie data
        """
        self.movies = movies_df.copy()
    
    def calculate_roi(self) -> pd.Series:
        """Calculate ROI (Return on Investment)"""
        budget = pd.to_numeric(self.movies['budget'], errors='coerce')
        revenue = pd.to_numeric(self.movies['revenue'], errors='coerce')
        return ((revenue - budget) / budget * 100).round(2)
    
    def calculate_profitability_ratio(self) -> pd.Series:
        """Calculate profitability ratio"""
        budget = pd.to_numeric(self.movies['budget'], errors='coerce')
        revenue = pd.to_numeric(self.movies['revenue'], errors='coerce')
        return (revenue / budget).round(2)
    
    def calculate_cost_per_rating_point(self) -> pd.Series:
        """Calculate cost per rating point"""
        budget = pd.to_numeric(self.movies['budget'], errors='coerce')
        rating = pd.to_numeric(self.movies['vote_average'], errors='coerce')
        return (budget / rating / 1e6).round(2)
    
    def calculate_revenue_per_minute(self) -> pd.Series:
        """Calculate revenue per minute"""
        revenue = pd.to_numeric(self.movies['revenue'], errors='coerce')
        runtime = pd.to_numeric(self.movies['runtime'], errors='coerce')
        return (revenue / runtime / 1e6).round(2)
    
    def efficiency_score(self) -> pd.Series:
        """
        Calculate combined efficiency score (0-100)
        Based on: ROI, Rating, Profitability
        """
        roi = self.calculate_roi()
        rating = pd.to_numeric(self.movies['vote_average'], errors='coerce')
        prof_ratio = self.calculate_profitability_ratio()
        
        # Normalize values
        roi_norm = (roi - roi.min()) / (roi.max() - roi.min()) * 100
        rating_norm = (rating - rating.min()) / (rating.max() - rating.min()) * 100
        prof_norm = (prof_ratio - prof_ratio.min()) / (prof_ratio.max() - prof_ratio.min()) * 100
        
        # Weighted average
        score = (roi_norm * 0.4 + rating_norm * 0.3 + prof_norm * 0.3).round(2)
        
        return score