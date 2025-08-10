# Used Car Data Scraper

A Python web scraper designed to collect used car listing data from Cars.com for machine learning analysis and price prediction models. This scraper supports both single-location and nationwide scraping to gather comprehensive datasets for data science projects.

## Overview

This scraper extracts key features from Cars.com used car listings including:

- Price, Year, Mileage
- Make, Model, Location
- Dealer vs Private seller information
- Dealer ratings
- Geographic scraping locations
- Automatically generated binary variables for ML models

## How It Works

### Core Components

1. **CarsComScraper Class**: Main scraper class that handles:
   - Session management with proper headers
   - URL building with search parameters (make, price, mileage, zip code)
   - Data extraction from Cars.com listing pages
   - Data cleaning and normalization
   - Nationwide scraping across 32+ major US cities

2. **Data Extraction Process**:

   ```python
   # 1. Build search URL with parameters
   url = self.build_search_url(page=page, zip_code=zip_code)
   
   # 2. Fetch and parse HTML content with rate limiting
   response = self.session.get(url)
   soup = BeautifulSoup(response.content, 'html.parser')
   
   # 3. Extract individual car listings
   listings = soup.find_all('div', class_='vehicle-card')
   
   # 4. Process each listing for relevant data
   for listing in listings:
       car_data = self.extract_car_details(listing)
   ```

3. **Data Processing Pipeline**:
   - Raw HTML → Structured data extraction
   - Text cleaning (prices, mileage, titles)
   - Geographic location tagging
   - Feature engineering (binary variables for ML)
   - Automatic duplicate handling

### Key Methods

- `build_search_url()`: Constructs Cars.com search URLs with filters
- `scrape_listing_page()`: Extracts all listings from a single page
- `extract_car_details()`: Parses individual vehicle listing data
- `scrape_multiple_pages()`: Single-location multi-page scraping
- `scrape_nationwide()`: Multi-city scraping across the US
- `get_us_major_zip_codes()`: Returns 32+ major US city zip codes
- `clean_price()` / `clean_mileage()`: Data normalization methods
- `create_binary_variables()`: Generates ML-ready binary features

## Installation & Setup

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:

   ```bash
   pip install requests beautifulsoup4 pandas python-dotenv
   ```

2. **Run the Scraper**:

   ```bash
   python scraper.py
   ```

## Usage

### Interactive Mode (Recommended)

The scraper includes an interactive main function that guides you through the process:

```python
python scraper.py
```

You'll be prompted to choose:

1. **Scraping Mode**: Single location vs. Nationwide
2. **Scale**: Number of pages or locations to scrape
3. **Output**: Automatic CSV generation with timestamps

### Nationwide Scraping

For comprehensive datasets, use nationwide mode which scrapes from 32+ major US cities:

```python
# Example: Nationwide scraping from 10 cities, 5 pages each
cars_data = scraper.scrape_nationwide(
    max_pages_per_location=5,
    delay_range=(3, 6),        # 3-6 seconds between cities
    request_delay=(1, 2),      # 1-2 seconds per request
    max_locations=10           # Limit to 10 cities
)
```

**Available Cities Include**:

- West Coast: Los Angeles, San Francisco, Seattle, Portland
- Southwest: Phoenix, Las Vegas, Denver, Austin, Dallas, Houston
- Midwest: Chicago, Detroit, Minneapolis, Kansas City
- Southeast: Miami, Atlanta, Nashville, Charlotte
- Northeast: New York, Boston, Philadelphia, Washington DC

### Single Location Scraping

For focused geographic analysis:

```python
# Example: Phoenix area, 20 pages
cars_data = scraper.scrape_multiple_pages(
    max_pages=20,
    delay_range=(2, 5),        # 2-5 seconds between pages
    request_delay=(0.5, 1.5)   # 0.5-1.5 seconds per request
)
```

### Getting Large Datasets (250+ Records)

The current scraper is designed to easily collect large datasets:

**Method 1: Nationwide Scraping**

```python
# This typically yields 500-2000+ records
cars_data = scraper.scrape_nationwide(
    max_pages_per_location=5,  # 5 pages per city
    max_locations=20           # 20 cities = 100 pages total
)
```

**Method 2: Single Location with Many Pages**

```python
# Phoenix area with extensive pagination
cars_data = scraper.scrape_multiple_pages(max_pages=50)
```

**Method 3: Multiple Manual Runs**

```python
# Run multiple times with different parameters
all_cars = []

# Different price ranges
for max_price in [20000, 40000, 60000, None]:
    url = scraper.build_search_url(max_price=max_price)
    cars = scraper.scrape_listing_page(url)
    all_cars.extend(cars)
```

### Data Output and Processing

The scraper automatically:

1. **Cleans Data**: Removes invalid entries, normalizes prices/mileage
2. **Creates Binary Variables**: For machine learning models
3. **Adds Location Data**: Geographic tagging for nationwide scrapes
4. **Saves with Timestamps**: Automatic CSV generation
5. **Provides Summary**: Dataset overview and statistics

## Configuration Options

### Search Parameters

The `build_search_url()` method supports these parameters:

```python
url = scraper.build_search_url(
    make="Toyota",          # Specific car make
    max_price=30000,        # Maximum price filter
    max_mileage=50000,      # Maximum mileage filter
    zip_code="85374",       # Geographic location
    page=1                  # Results page number
)
```

### Rate Limiting and Delays

Built-in rate limiting prevents blocking:

```python
# Nationwide scraping delays
scraper.scrape_nationwide(
    delay_range=(3, 6),     # 3-6 seconds between cities
    request_delay=(1, 2),   # 1-2 seconds per request
    max_pages_per_location=5
)

# Single location delays
scraper.scrape_multiple_pages(
    delay_range=(2, 5),     # 2-5 seconds between pages
    request_delay=(0.5, 1.5)  # 0.5-1.5 seconds per request
)
```

## Features for ML Models

The scraper automatically generates binary variables optimized for machine learning:

- `high_priced`: Above median price (classification target)
- `low_mileage`: Less than 50,000 miles
- `is_luxury`: Luxury brand (BMW, Mercedes-Benz, Audi, Lexus, Infiniti, Acura)
- `is_recent`: 2020 or newer model year
- `is_dealer`: Dealer vs private sale
- `scrape_location`: Geographic origin (for nationwide scrapes)
- `scrape_zip`: ZIP code of search location

## Example Workflows

### Quick Start - Small Dataset

```python
from scraper import CarsComScraper

# Initialize and run interactively
scraper = CarsComScraper()
# Run: python scraper.py
# Choose: Single location, 10 pages
```

### Large Dataset - Nationwide

```python
# Programmatic usage for nationwide scraping
scraper = CarsComScraper()

# Scrape from 15 major cities, 5 pages each
cars_data = scraper.scrape_nationwide(
    max_pages_per_location=5,
    max_locations=15,
    delay_range=(3, 6),
    request_delay=(1, 2)
)

# Convert to DataFrame with all features
df = pd.DataFrame(cars_data)
df = df.dropna(subset=['price', 'mileage'])
df = scraper.create_binary_variables(df)

# Save with timestamp
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
df.to_csv(f'used_cars_dataset_{timestamp}.csv', index=False)
```

### Custom Analysis

```python
# Focus on specific regions or criteria
west_coast_cities = ['Los Angeles, CA', 'San Francisco, CA', 'Seattle, WA']

# Custom zip code selection
custom_zips = {'Los Angeles': '90210', 'Phoenix': '85001', 'Chicago': '60601'}

for city, zip_code in custom_zips.items():
    cars = scraper.scrape_multiple_pages(max_pages=10)
    # Process city-specific data
```

## Troubleshooting

### Common Issues

1. **Low Record Count**:
   - Increase `max_pages` parameter (try 20-50 pages)
   - Use nationwide mode for geographic diversity
   - Remove price/mileage filters from search URL
   - Try different zip codes or time periods

2. **Missing Data Fields**:
   - Check Cars.com website for CSS class changes
   - Verify network connectivity and response codes
   - Add error handling and logging for debugging

3. **Rate Limiting/Blocking**:
   - Increase delays between requests (`request_delay`)
   - Add longer delays between pages (`delay_range`)
   - Use different User-Agent strings
   - Implement proxy rotation if needed

4. **Data Quality Issues**:
   - Review data cleaning functions
   - Check for duplicate entries
   - Validate price and mileage ranges
   - Ensure proper year parsing

### Performance Tips

- **Optimal Settings**: Start with 5 pages per location for testing
- **Rate Limiting**: Use 1-2 second delays between requests
- **Error Recovery**: Save intermediate results to avoid data loss
- **Monitoring**: Watch console output for success/failure rates
- **Estimation**: ~50-100 cars per page, ~500-1000 cars per city

## Data Output

The final CSV dataset includes these columns:

- **Basic Information**: price, year, make, model, mileage, location
- **Dealer Information**: dealer_name, is_dealer, dealer_rating  
- **Geographic Data**: scrape_location, scrape_zip (nationwide mode)
- **ML Features**: high_priced, low_mileage, is_luxury, is_recent

This scraper provides a comprehensive dataset for machine learning projects analyzing used car pricing, market trends, and geographic variations in the automotive market.
