import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """
    Utility class để load và xử lý dữ liệu phim từ các file CSV
    """
    
    def __init__(self, 
                 movies_path: str = '/Users/phuongdoan/Downloads/dataset/movies_metadata.csv',
                 credits_path: str = '/Users/phuongdoan/Downloads/dataset/credits.csv',
                 keywords_path: str = '/Users/phuongdoan/Downloads/dataset/keywords.csv',
                 ratings_path: str = '/Users/phuongdoan/Downloads/dataset/ratings.csv'):
        """
        Khởi tạo DataLoader với các đường dẫn file
        
        Args:
            movies_path: file path of movies_metadata.csv
            credits_path: file path of credits.csv
            keywords_path: file path of keywords.csv
            ratings_path: file path of ratings.csv
        """
        self.movies_path = movies_path
        self.credits_path = credits_path
        self.keywords_path = keywords_path
        self.ratings_path = ratings_path
        
        self.movies = None
        self.credits = None
        self.keywords = None
        self.ratings = None
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load tất cả các file dữ liệu
        
        Returns:
            Tuple(movies_df, credits_df, keywords_df, ratings_df)
        """
        print("🔄 Đang load dữ liệu...")
        
        self.movies = self.load_movies()
        self.credits = self.load_credits()
        self.keywords = self.load_keywords()
        self.ratings = self.load_ratings()
        
        print("✅ Load dữ liệu thành công!")
        return self.movies, self.credits, self.keywords, self.ratings
    
    def load_movies(self) -> pd.DataFrame:
        """Load movies_metadata.csv"""
        print(f"   📥 Loading movies from: {self.movies_path}")
        
        df = pd.read_csv(self.movies_path, low_memory=False)
        print(f"   ✓ Movies: {len(df)} dòng, {len(df.columns)} cột")
        
        return df
    
    def load_credits(self) -> pd.DataFrame:
        """Load credits.csv"""
        print(f"   📥 Loading credits from: {self.credits_path}")
        
        df = pd.read_csv(self.credits_path)
        
        # Xử lý cast và crew
        df['cast_names'] = df['cast'].apply(self._extract_names)
        df['crew_names'] = df['crew'].apply(self._extract_names)
        
        print(f"   ✓ Credits: {len(df)} dòng, {len(df.columns)} cột")
        
        return df
    
    def load_keywords(self) -> pd.DataFrame:
        """Load keywords.csv"""
        print(f"   📥 Loading keywords from: {self.keywords_path}")
        
        df = pd.read_csv(self.keywords_path)
        
        # Xử lý keywords
        df['keywords'] = df['keywords'].apply(self._extract_keywords)
        
        print(f"   ✓ Keywords: {len(df)} dòng, {len(df.columns)} cột")
        
        return df
    
    def load_ratings(self) -> pd.DataFrame:
        """Load ratings.csv"""
        print(f"   📥 Loading ratings from: {self.ratings_path}")
        
        df = pd.read_csv(self.ratings_path)
        print(f"   ✓ Ratings: {len(df)} dòng, {len(df.columns)} cột")
        
        return df
    
    @staticmethod
    def _extract_names(data_str) -> str:
        """
        Extract names từ JSON string (cast hoặc crew)
        
        Args:
            data_str: JSON string chứa danh sách cast/crew
            
        Returns:
            String chứa các tên cách nhau bằng dấu phẩy
        """
        import json
        
        try:
            if isinstance(data_str, str):
                data = json.loads(data_str.replace("'", '"'))
                names = [person.get('name', '') for person in data if 'name' in person]
                return ', '.join(names[:5])  # Lấy tối đa 5 người
        except:
            pass
        
        return ''
    
    @staticmethod
    def _extract_keywords(keywords_str) -> str:
        """
        Extract keywords từ JSON string
        
        Args:
            keywords_str: JSON string chứa danh sách keywords
            
        Returns:
            String chứa các keywords cách nhau bằng dấu phẩy
        """
        import json
        
        try:
            if isinstance(keywords_str, str):
                data = json.loads(keywords_str.replace("'", '"'))
                keywords = [keyword.get('name', '') for keyword in data if 'name' in keyword]
                return ', '.join(keywords[:10])  # Lấy tối đa 10 keywords
        except:
            pass
        
        return ''
    
    def get_data_info(self):
        """In thông tin chi tiết về các DataFrame"""
        if self.movies is not None:
            print("\n" + "="*80)
            print("MOVIES DATAFRAME INFO")
            print("="*80)
            self.movies.info()
        
        if self.credits is not None:
            print("\n" + "="*80)
            print("CREDITS DATAFRAME INFO")
            print("="*80)
            self.credits.info()
        
        if self.keywords is not None:
            print("\n" + "="*80)
            print("KEYWORDS DATAFRAME INFO")
            print("="*80)
            self.keywords.info()
        
        if self.ratings is not None:
            print("\n" + "="*80)
            print("RATINGS DATAFRAME INFO")
            print("="*80)
            print(self.ratings.info())
    
    def get_basic_stats(self) -> dict:
        """
        Lấy thống kê cơ bản về dữ liệu
        
        Returns:
            Dictionary chứa các thống kê
        """
        stats = {
            'total_movies': len(self.movies) if self.movies is not None else 0,
            'total_ratings': len(self.ratings) if self.ratings is not None else 0,
            'unique_users': self.ratings['userId'].nunique() if self.ratings is not None else 0,
            'rating_range': (
                self.ratings['rating'].min(),
                self.ratings['rating'].max()
            ) if self.ratings is not None else (0, 0),
        }
        
        return stats


def validate_data(movies_df: pd.DataFrame, credits_df: pd.DataFrame, 
                 keywords_df: pd.DataFrame, ratings_df: pd.DataFrame) -> bool:
    """
    Kiểm tra tính hợp lệ của dữ liệu
    
    Args:
        movies_df: Movies DataFrame
        credits_df: Credits DataFrame
        keywords_df: Keywords DataFrame
        ratings_df: Ratings DataFrame
        
    Returns:
        True nếu dữ liệu hợp lệ, False nếu không
    """
    checks = []
    
    # Kiểm tra movies
    if 'id' not in movies_df.columns:
        print("❌ Movies: Thiếu cột 'id'")
        checks.append(False)
    else:
        checks.append(True)
    
    # Kiểm tra credits
    if 'id' not in credits_df.columns:
        print("❌ Credits: Thiếu cột 'id'")
        checks.append(False)
    else:
        checks.append(True)
    
    # Kiểm tra keywords
    if 'id' not in keywords_df.columns:
        print("❌ Keywords: Thiếu cột 'id'")
        checks.append(False)
    else:
        checks.append(True)
    
    # Kiểm tra ratings
    if 'userId' not in ratings_df.columns or 'movieId' not in ratings_df.columns:
        print("❌ Ratings: Thiếu cột 'userId' hoặc 'movieId'")
        checks.append(False)
    else:
        checks.append(True)
    
    if all(checks):
        print("✅ Dữ liệu hợp lệ!")
        return True
    
    return False

def create_data_summary(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo tóm tắt dữ liệu về phim
    
    Args:
        movies_df: Movies DataFrame
        
    Returns:
        DataFrame chứa tóm tắt
    """
    summary = pd.DataFrame({
        'Metric': [
            'Total Movies',
            'Movies with Budget',
            'Movies with Revenue',
            'Movies with Rating',
            'Average Budget (M$)',
            'Average Revenue (M$)',
            'Average Rating',
            'Rating Range'
        ],
        'Value': [
            len(movies_df),
            len(movies_df[movies_df['budget'].astype(str).ne('0')]),
            len(movies_df[movies_df['revenue'].astype(str).ne('0')]),
            len(movies_df[movies_df['vote_average'].notna()]),
            f"{pd.to_numeric(movies_df['budget'], errors='coerce').mean() / 1e6:.2f}",
            f"{pd.to_numeric(movies_df['revenue'], errors='coerce').mean() / 1e6:.2f}",
            f"{movies_df['vote_average'].astype(float).mean():.2f}",
            f"{movies_df['vote_average'].astype(float).min():.1f} - {movies_df['vote_average'].astype(float).max():.1f}"
        ]
    })
    
    return summary

def quick_load():
    """
    Cách sử dụng đơn giản nhất
    
    Returns:
        Tuple(movies, credits, keywords, ratings)
    """
    loader = DataLoader()
    movies, credits, keywords, ratings = loader.load_all_data()
    validate_data(movies, credits, keywords, ratings)
    
    return movies, credits, keywords, ratings
