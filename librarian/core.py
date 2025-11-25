import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json

import warnings
warnings.filterwarnings('ignore')


class MoviesAnalytics:
    """
    Analyzing factors affecting movies revenue and ratings
    """
    
    def __init__(self, movies, credits, keywords, ratings):
        """
        Start the analytics system
        Args:
            movies: DataFrame contains movies
            credits: DataFrame contains cast and crew
            keywords: DataFrame contains keywords of movies
            ratings: DataFrame contains ratings from users
        """
        self.movies = movies.copy()
        self.credits = credits.copy()
        self.keywords = keywords.copy()
        self.ratings = ratings.copy()
        self._pre_data()

    def _pre_data(self):
        """Prepare and process data"""
        # Convert data types
        self.movies['id'] = pd.to_numeric(self.movies['id'], errors='coerce')
        self.movies['budget'] = pd.to_numeric(self.movies['budget'], errors='coerce')
        self.movies['revenue'] = pd.to_numeric(self.movies['revenue'], errors='coerce')
        self.movies['runtime'] = pd.to_numeric(self.movies['runtime'], errors='coerce')
        self.movies['release_date'] = pd.to_datetime(self.movies['release_date'], errors='coerce')
        self.movies['popularity'] = pd.to_numeric(self.movies['popularity'], errors='coerce')
        self.movies['vote_average'] = pd.to_numeric(self.movies['vote_average'], errors='coerce')
        self.movies['vote_count'] = pd.to_numeric(self.movies['vote_count'], errors='coerce')

        # Add release year and month
        self.movies['release_year'] = self.movies['release_date'].dt.year
        self.movies['release_month'] = self.movies['release_date'].dt.month
        
        # Calculate profit and ROI
        self.movies['profit'] = self.movies['revenue'] - self.movies['budget']
        self.movies['roi'] = np.where(
            self.movies['budget'] > 0,
            ((self.movies['revenue'] - self.movies['budget']) / self.movies['budget']) * 100,
            np.nan
        )

        # Keep only movies with sufficient data
        self.movies_clean = self.movies[
            (self.movies['budget'] > 0) &
            (self.movies['revenue'] > 0) &
            (self.movies['runtime'] > 0) &
            (self.movies['release_year'].notna())
        ].copy()
        
        # Process genres (parse JSON)
        self.movies_clean['genres_list'] = self.movies_clean['genres'].apply(
            self._parse_json_safe
        )
        
        # Explode genres into separate rows for analytics
        self.movies_exploded = self.movies_clean.explode('genres_list')
    
    def _parse_json_safe(self, x):
        """Safely parse JSON strings"""
        if isinstance(x, str):
            try:
                data = json.loads(x.replace("'", '"'))
                if isinstance(data, list):
                    return [item.get('name', '') if isinstance(item, dict) else str(item) for item in data]
            except:
                pass
        return []
    
    def analyze_genre_impact(self) -> Dict:
        """
        Analyzing the impact of genres on revenue and ratings
        
        Returns:
            Dictionary containing statistics by genre
        """
        genre_stats = self.movies_exploded.groupby('genres_list').agg({
            'revenue': ['mean', 'median', 'sum'],
            'budget': ['mean', 'median'],
            'vote_average': 'mean',
            'roi': 'mean',
            'profit': 'mean',
            'runtime': 'mean',
            'id': 'count'
        }).round(2)
        
        genre_stats.columns = ['avg_revenue', 'median_revenue', 'total_revenue', 
                               'avg_budget', 'median_budget', 'avg_rating', 
                               'avg_roi', 'avg_profit', 'avg_runtime', 'movie_count']
        
        return genre_stats.sort_values('avg_revenue', ascending=False)
    
    def analyze_budget_revenue_relationship(self) -> Tuple[float, pd.DataFrame]:
        """
        Analyze the relationship between budget and revenue
        
        Returns:
            Tuple (correlation coefficient, grouped statistics)
        """
        correlation = self.movies_clean[['budget', 'revenue']].corr().iloc[0, 1]
        
        # Divide budget into bins
        budget_bins = pd.cut(self.movies_clean['budget'], bins=10)
        budget_analysis = self.movies_clean.groupby(budget_bins, observed=True).agg({
            'revenue': ['mean', 'median', 'min', 'max'],
            'roi': 'mean',
            'vote_average': 'mean',
            'id': 'count'
        }).round(2)
        
        return correlation, budget_analysis
    
    def analyze_runtime_impact(self) -> pd.DataFrame:
        """
        Analyze the impact of runtime on revenue and ratings
        
        Returns:
            DataFrame containing statistics by runtime
        """
        runtime_bins = pd.cut(self.movies_clean['runtime'], bins=8)
        runtime_analysis = self.movies_clean.groupby(runtime_bins, observed=True).agg({
            'revenue': 'mean',
            'budget': 'mean',
            'vote_average': 'mean',
            'roi': 'mean',
            'id': 'count'
        }).round(2)
        runtime_analysis.columns = ['avg_revenue', 'avg_budget', 'avg_rating', 'avg_roi', 'count']
        
        return runtime_analysis
    
    def analyze_seasonal_release_impact(self) -> pd.DataFrame:
        """
        Analyzing the impact of release season on revenue
        
        Returns:
            DataFrame containing the statistics by release month
        """
        seasonal_analysis = self.movies_clean.groupby('release_month').agg({
            'revenue': ['mean', 'median', 'count'],
            'vote_average': 'mean',
            'roi': 'mean'
        }).round(2)
        
        seasonal_analysis.columns = ['avg_revenue', 'median_revenue', 'movie_count', 'avg_rating', 'avg_roi']
        
        return seasonal_analysis
    
    def analyze_industry_trends(self) -> pd.DataFrame:
        """
        Analyzing movie industry trends over years
        
        Returns:
            DataFrame containing yearly trends
        """
        yearly_stats = self.movies_clean.groupby('release_year').agg({
            'budget': ['mean', 'median', 'sum'],
            'revenue': ['mean', 'median', 'sum'],
            'profit': 'mean',
            'roi': 'mean',
            'vote_average': 'mean',
            'runtime': 'mean',
            'id': 'count'
        }).round(2)
        
        yearly_stats.columns = ['avg_budget', 'median_budget', 'total_budget',
                                'avg_revenue', 'median_revenue', 'total_revenue',
                                'avg_profit', 'avg_roi', 'avg_rating', 'avg_runtime', 'movie_count']
        
        return yearly_stats
    
    def plot_genre_revenue_comparison(self, top_n: int = 10):
        """Bar chart comparing revenue by genre"""
        genre_stats = self.analyze_genre_impact()
        top_genres = genre_stats.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        genres = top_genres.index
        revenues = top_genres['avg_revenue'] / 1e6
        
        bars = ax.barh(genres, revenues, color=plt.cm.viridis(np.linspace(0, 1, len(genres))))
        ax.set_xlabel('Average Revenue (Million USD)', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Movie genres by average revenue', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'${width:.0f}M', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_budget_vs_revenue(self):
        """Scatter plot: Budget vs Revenue"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        scatter = ax.scatter(self.movies_clean['budget']/1e6, 
                            self.movies_clean['revenue']/1e6,
                            c=self.movies_clean['vote_average'],
                            s=self.movies_clean['vote_count']/100,
                            alpha=0.6, cmap='coolwarm', edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(self.movies_clean['budget']/1e6, self.movies_clean['revenue']/1e6, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(self.movies_clean['budget'].min()/1e6, 
                             self.movies_clean['budget'].max()/1e6, 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend line')
        
        ax.set_xlabel('Budget (Million USD)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Revenue (Million USD)', fontsize=12, fontweight='bold')
        ax.set_title('Relationship between budget and revenue\n(Size = Vote count, Color = Rating)', 
                    fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Average rating', fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_profitability_trends(self):
        """Profitability trends over the years"""
        yearly_stats = self.analyze_industry_trends()
        yearly_stats = yearly_stats[yearly_stats.index >= 2000]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Profit chart
        ax1.plot(yearly_stats.index, yearly_stats['avg_profit']/1e6, 
                marker='o', linewidth=2.5, markersize=8, color='#2ecc71', label='Profit')
        ax1.fill_between(yearly_stats.index, yearly_stats['avg_profit']/1e6, alpha=0.3, color='#2ecc71')
        ax1.set_ylabel('Average profit (Million USD)', fontsize=11, fontweight='bold')
        ax1.set_title('Movie profit trends over the years', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # ROI chart
        ax2.plot(yearly_stats.index, yearly_stats['avg_roi'], 
                marker='s', linewidth=2.5, markersize=8, color='#e74c3c', label='ROI (%)')
        ax2.fill_between(yearly_stats.index, yearly_stats['avg_roi'], alpha=0.3, color='#e74c3c')
        ax2.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Average ROI (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Movie ROI trends over the years', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_runtime_impact(self):
        """Impact of runtime on revenue and ratings"""
        runtime_analysis = self.analyze_runtime_impact()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Revenue by runtime
        x_labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in runtime_analysis.index]
        ax1.bar(range(len(runtime_analysis)), runtime_analysis['avg_revenue']/1e6, 
               color=plt.cm.Blues(np.linspace(0.4, 0.9, len(runtime_analysis))))
        ax1.set_xticks(range(len(runtime_analysis)))
        ax1.set_xticklabels(x_labels, rotation=45)
        ax1.set_ylabel('Average revenue (Million USD)', fontsize=11, fontweight='bold')
        ax1.set_title('Revenue by movie runtime', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Rating by runtime
        ax2.plot(range(len(runtime_analysis)), runtime_analysis['avg_rating'], 
                marker='o', linewidth=2.5, markersize=8, color='#3498db', label='Rating')
        ax2.set_xticks(range(len(runtime_analysis)))
        ax2.set_xticklabels(x_labels, rotation=45)
        ax2.set_ylabel('Average rating', fontsize=11, fontweight='bold')
        ax2.set_title('Rating by movie runtime', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_seasonal_release_patterns(self):
        """Seasonal release patterns"""
        seasonal_analysis = self.analyze_seasonal_release_impact()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Revenue by month
        ax1.bar(seasonal_analysis.index, seasonal_analysis['avg_revenue']/1e6,
               color=plt.cm.RdYlGn(np.linspace(0.3, 0.7, 12)))
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(months)
        ax1.set_ylabel('Average revenue (Million USD)', fontsize=11, fontweight='bold')
        ax1.set_title('Revenue by release month', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Rating by month
        ax2.plot(seasonal_analysis.index, seasonal_analysis['avg_rating'], 
                marker='D', linewidth=2.5, markersize=8, color='#9b59b6')
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(months)
        ax2.set_ylabel('Average rating', fontsize=11, fontweight='bold')
        ax2.set_title('Rating by release month', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_industry_evolution(self):
        """Movie industry evolution over time"""
        yearly_stats = self.analyze_industry_trends()
        yearly_stats = yearly_stats[yearly_stats.index >= 2000]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Budget trends
        ax1.plot(yearly_stats.index, yearly_stats['avg_budget']/1e6, 
                marker='o', color='#e74c3c', linewidth=2.5, markersize=6)
        ax1.set_ylabel('Average budget (Million USD)', fontweight='bold')
        ax1.set_title('Movie budget trends', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Revenue trends
        ax2.plot(yearly_stats.index, yearly_stats['avg_revenue']/1e6, 
                marker='s', color='#27ae60', linewidth=2.5, markersize=6)
        ax2.set_ylabel('Average revenue (Million USD)', fontweight='bold')
        ax2.set_title('Movie revenue trends', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Rating trends
        ax3.plot(yearly_stats.index, yearly_stats['avg_rating'], 
                marker='^', color='#3498db', linewidth=2.5, markersize=6)
        ax3.set_xlabel('Year', fontweight='bold')
        ax3.set_ylabel('Average rating', fontweight='bold')
        ax3.set_title('Movie rating trends', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Movie count trends
        ax4.bar(yearly_stats.index, yearly_stats['movie_count'], 
               color=plt.cm.Spectral(np.linspace(0, 1, len(yearly_stats))))
        ax4.set_xlabel('Year', fontweight='bold')
        ax4.set_ylabel('Number of movies', fontweight='bold')
        ax4.set_title('Movie releases trends', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self) -> str:
        """
        Generate comprehensive analysis report
        
        Returns:
            String containing the report
        """
        budget_corr, _ = self.analyze_budget_revenue_relationship()
        genre_stats = self.analyze_genre_impact()
        seasonal = self.analyze_seasonal_release_impact()
        
        report = f"""

- GENERAL STATISTICS:
   • Total movies analyzed: {len(self.movies_clean):,}
   • Release year range: {int(self.movies_clean['release_year'].min())} - {int(self.movies_clean['release_year'].max())}
   • Average revenue: ${self.movies_clean['revenue'].mean()/1e6:.2f}M
   • Average budget: ${self.movies_clean['budget'].mean()/1e6:.2f}M
   • Average rating: {self.movies_clean['vote_average'].mean():.2f}/10

- BUDGET VS REVENUE ANALYSIS:
   • Correlation: {budget_corr:.3f}
   • Average profit: ${self.movies_clean['profit'].mean()/1e6:.2f}M
   • Average ROI: {self.movies_clean['roi'].mean():.2f}%

- TOP 5 GENRES BY REVENUE:
{self._format_genre_stats(genre_stats.head(5))}

- BEST RELEASE MONTH:
   Month {int(seasonal['avg_revenue'].idxmax())}: ${seasonal['avg_revenue'].max()/1e6:.2f}M average revenue
   Rating: {seasonal.loc[seasonal['avg_revenue'].idxmax(), 'avg_rating']:.2f}

- OPTIMAL MOVIE RUNTIME:
   Average: {self.movies_clean['runtime'].mean():.0f} minutes
   Range: {self.movies_clean['runtime'].min():.0f} - {self.movies_clean['runtime'].max():.0f} minutes
"""
        return report
    
    def _format_genre_stats(self, genre_stats_df) -> str:
        """Format genre statistics for display"""
        result = ""
        for i, (genre, row) in enumerate(genre_stats_df.iterrows(), 1):
            result += f"   {i}. {genre}: ${row['avg_revenue']/1e6:.2f}M | Rating: {row['avg_rating']:.2f} | Count: {int(row['movie_count'])}\n"
        return result