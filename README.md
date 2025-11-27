
# Python Project - Anatomy of a Blockbuster
##### by Duy My Phuong Doan
### Introduction to the Project
- This is an individual project analyzing fators affecting movie revenue and ratings.
- Course: Coding for Data Science and Data Management.
- Instructor: Professor Sergio Picascia.
- Project goal: Analyze how genre, budget, runtime and release data connect to box office revenue and ratings, and investigate how the film industry as changed over the years.
- Dataset: The Movie Dataset with 45,466 movies including:
  - Movie metadat (budget, revenue, runtime, release date, genres, ratings)
  - Credits (Cast and Crew information)
  - Keywords 
  - Users ratings
  - Link to the dataset: [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
### Project Objective
#### Primary analysis goal
1. Identify revenue drivers
- Analyze the relationship between budget and revenue.
- Determine which genres generate the highest revenue.
- Study the impact of runtime on box office performance.
2. Understand rating patterns
- Explore how different factors affect user ratings.
- Compare high-rated vs low-rated movies.
- Indentify seasonal trends in audience reception.
3. Industry evolution tracking
- Examine how movie budgets and revenues have changed over time.
- Identify emerging trends in the film industry.
- Analyze profitability trends (ROI and profit margins).
4. Segment and profile movies
- Segment movies by budget, rating, and runtime.
- Create performance metrics (ROI, efficiency score).
- Identify best-performing movie categories.

### Methodology
#### Key analytical approaches:
- Correlation analysis: examine relationships between variables (budget, revenue, runtime, ratings).
- Segmentation analysis: divide movies into groups based on budget, rating, and runtime.
- Trend analysis: track changes over time using moving averages and year-over-year growth
- Performance metrics: calculate ROI, profitability ratio and efficiency metrics
- Statistical testing: detect outliers and test hypotheses using IQR and Z-score methods
#### Key formulas:
- ROI(return on investment): [(Revenue - Budget)/ Budget] x  100.
- Profitability ration: Revenue / Budget
- Efficiency Score: (ROI x 0.4 + Rating x 0.3 + Profitability x 0.3) (normalized 0-100)


### Technology Stack
Language: Python

