import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from urllib.parse import urljoin, quote

class CarsComScraper:
    def __init__(self):
        self.base_url = "https://www.cars.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def build_search_url(self, make=None, max_price=None, max_mileage=None, 
                        zip_code="85374", page=1):
        """Build search URL for Cars.com"""
        params = []
        params.append(f"page={page}")
        params.append(f"zip={zip_code}")
        params.append("listingTypes=USED")
        
        if make:
            params.append(f"makes[]={quote(make)}")
        if max_price:
            params.append(f"priceMax={max_price}")
        if max_mileage:
            params.append(f"mileageMax={max_mileage}")
            
        search_url = f"{self.base_url}/shopping/results/?" + "&".join(params)
        return search_url
    
    def scrape_listing_page(self, url, request_delay=(0.5, 1.5)):
        """Scrape individual car listings from a search results page"""
        try:
            # Add delay before making request to prevent rate limiting
            delay = random.uniform(request_delay[0], request_delay[1])
            time.sleep(delay)
            
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all car listing containers
            listings = soup.find_all('div', class_='vehicle-card')
            
            cars_data = []
            for listing in listings:
                car_data = self.extract_car_details(listing)
                if car_data:
                    cars_data.append(car_data)
                    
            return cars_data
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return []
    
    def extract_car_details(self, listing_element):
        """Extract details from individual car listing"""
        try:
            car_data = {}
            
            # Price
            price_elem = listing_element.find('span', class_='primary-price')
            if price_elem:
                price_text = price_elem.get_text(strip=True)
                car_data['price'] = self.clean_price(price_text)
            
            # Title (includes year, make, model)
            title_elem = listing_element.find('h2', class_='title')
            if title_elem:
                title = title_elem.get_text(strip=True)
                car_data.update(self.parse_title(title))
            
            # Mileage
            mileage_elem = listing_element.find('div', class_='mileage')
            if mileage_elem:
                mileage_text = mileage_elem.get_text(strip=True)
                car_data['mileage'] = self.clean_mileage(mileage_text)
            
            # Location
            location_elem = listing_element.find('div', class_='miles-from')
            if location_elem:
                car_data['location'] = location_elem.get_text(strip=True)
            
            # Dealer info
            dealer_elem = listing_element.find('div', class_='dealer-name')
            if dealer_elem:
                car_data['dealer_name'] = dealer_elem.get_text(strip=True)
                car_data['is_dealer'] = 1
            else:
                car_data['is_dealer'] = 0
            
            # Rating
            rating_elem = listing_element.find('span', class_='sds-rating__count')
            if rating_elem:
                car_data['dealer_rating'] = rating_elem.get_text(strip=True)
            
            return car_data
            
        except Exception as e:
            print(f"Error extracting car details: {e}")
            return None
    
    def clean_price(self, price_text):
        """Clean and convert price to numeric"""
        try:
            # Remove $ and commas, convert to int
            price = price_text.replace('$', '').replace(',', '')
            return int(price) if price.isdigit() else None
        except:
            return None
    
    def clean_mileage(self, mileage_text):
        """Clean and convert mileage to numeric"""
        try:
            # Extract numbers from mileage text
            mileage = ''.join(filter(str.isdigit, mileage_text))
            return int(mileage) if mileage else None
        except:
            return None
    
    def parse_title(self, title):
        """Parse year, make, model from title"""
        try:
            parts = title.split()
            data = {}
            
            # First part is usually the year
            if parts and parts[0].isdigit():
                data['year'] = int(parts[0])
                
            # Second part is usually the make
            if len(parts) > 1:
                data['make'] = parts[1]
                
            # Rest is model
            if len(parts) > 2:
                data['model'] = ' '.join(parts[2:])
                
            return data
        except:
            return {}
    
    def scrape_multiple_pages(self, max_pages=5, delay_range=(2, 5), request_delay=(0.5, 1.5)):
        """Scrape multiple pages of results"""
        all_cars = []
        
        for page in range(1, max_pages + 1):
            print(f"Scraping page {page}...")
            
            # Build URL for current page
            url = self.build_search_url(page=page)
            
            # Scrape the page with request delay
            cars = self.scrape_listing_page(url, request_delay=request_delay)
            all_cars.extend(cars)
            
            print(f"Found {len(cars)} cars on page {page}")
            
            # Random delay between page requests
            if page < max_pages:
                delay = random.uniform(delay_range[0], delay_range[1])
                print(f"Waiting {delay:.1f} seconds before next page...")
                time.sleep(delay)
        
        return all_cars
    
    def get_us_major_zip_codes(self):
        """Get major zip codes from across the United States"""
        # Major cities across different regions of the US
        us_zip_codes = {
            # West Coast
            'Los Angeles, CA': '90210',
            'San Francisco, CA': '94102',
            'Seattle, WA': '98101',
            'Portland, OR': '97201',
            'San Diego, CA': '92101',
            
            # Southwest
            'Phoenix, AZ': '85001',
            'Las Vegas, NV': '89101',
            'Denver, CO': '80202',
            'Austin, TX': '78701',
            'Dallas, TX': '75201',
            'Houston, TX': '77002',
            'San Antonio, TX': '78205',
            
            # Midwest
            'Chicago, IL': '60601',
            'Detroit, MI': '48201',
            'Minneapolis, MN': '55401',
            'Kansas City, MO': '64108',
            'St. Louis, MO': '63101',
            'Milwaukee, WI': '53202',
            'Cleveland, OH': '44113',
            'Indianapolis, IN': '46204',
            
            # Southeast
            'Miami, FL': '33101',
            'Tampa, FL': '33602',
            'Orlando, FL': '32801',
            'Atlanta, GA': '30309',
            'Nashville, TN': '37201',
            'Charlotte, NC': '28202',
            'New Orleans, LA': '70112',
            'Birmingham, AL': '35203',
            
            # Northeast
            'New York, NY': '10001',
            'Boston, MA': '02101',
            'Philadelphia, PA': '19102',
            'Washington, DC': '20001',
            'Baltimore, MD': '21201',
            'Pittsburgh, PA': '15222',
            'Buffalo, NY': '14202'
        }
        
        return us_zip_codes
    
    def scrape_nationwide(self, max_pages_per_location=5, delay_range=(3, 6), 
                         request_delay=(1, 2), max_locations=None):
        """Scrape car data from major cities across the United States"""
        
        zip_codes = self.get_us_major_zip_codes()
        
        # Limit locations if specified
        if max_locations:
            zip_codes = dict(list(zip_codes.items())[:max_locations])
        
        all_cars_nationwide = []
        total_locations = len(zip_codes)
        
        print(f"Starting nationwide scrape from {total_locations} locations...")
        print(f"Estimated time: {(total_locations * max_pages_per_location * 3) / 60:.1f} minutes")
        
        for i, (city, zip_code) in enumerate(zip_codes.items(), 1):
            print(f"\n--- Scraping {city} ({i}/{total_locations}) ---")
            
            try:
                # Scrape multiple pages for this location
                location_cars = []
                
                for page in range(1, max_pages_per_location + 1):
                    print(f"  Page {page}/{max_pages_per_location} for {city}...")
                    
                    # Build URL for current page and location
                    url = self.build_search_url(zip_code=zip_code, page=page)
                    
                    # Scrape the page with request delay
                    cars = self.scrape_listing_page(url, request_delay=request_delay)
                    
                    # Add location info to each car
                    for car in cars:
                        car['scrape_location'] = city
                        car['scrape_zip'] = zip_code
                    
                    location_cars.extend(cars)
                    print(f"    Found {len(cars)} cars on page {page}")
                    
                    # Delay between pages within same location
                    if page < max_pages_per_location:
                        page_delay = random.uniform(1, 2)
                        time.sleep(page_delay)
                
                all_cars_nationwide.extend(location_cars)
                print(f"  Total from {city}: {len(location_cars)} cars")
                
                # Longer delay between different locations
                if i < total_locations:
                    location_delay = random.uniform(delay_range[0], delay_range[1])
                    print(f"  Waiting {location_delay:.1f} seconds before next location...")
                    time.sleep(location_delay)
                    
            except Exception as e:
                print(f"  Error scraping {city}: {e}")
                continue
        
        print(f"\n=== Nationwide scrape complete ===")
        print(f"Total cars scraped: {len(all_cars_nationwide)}")
        
        return all_cars_nationwide
    
    def create_binary_variables(self, df):
        """Create binary variables for classification models"""
        
        # High-priced vehicle (above median)
        if 'price' in df.columns:
            median_price = df['price'].median()
            df['high_priced'] = (df['price'] > median_price).astype(int)
        
        # Low mileage (less than 50k miles)
        if 'mileage' in df.columns:
            df['low_mileage'] = (df['mileage'] < 50000).astype(int)
        
        # Luxury brand
        luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Infiniti', 'Acura']
        if 'make' in df.columns:
            df['is_luxury'] = df['make'].isin(luxury_brands).astype(int)
        
        # Recent model (2020 or newer)
        if 'year' in df.columns:
            df['is_recent'] = (df['year'] >= 2020).astype(int)
        
        return df

# Usage example
def main():
    scraper = CarsComScraper()
    
    # Choose scraping mode
    nationwide = input("Scrape nationwide? (y/n): ").lower().startswith('y')
    
    if nationwide:
        # Nationwide scraping
        print("Starting nationwide scrape...")
        max_locations = input("Max locations (press Enter for all 32): ")
        max_locations = int(max_locations) if max_locations.isdigit() else None
        
        cars_data = scraper.scrape_nationwide(
            max_pages_per_location=5,  # Conservative for nationwide
            delay_range=(3, 6),        # 3-6 seconds between locations
            request_delay=(1, 2),      # 1-2 seconds per request
            max_locations=max_locations
        )
    else:
        # Single location scraping (original method)
        print("Starting single location scrape...")
        max_pages = int(input("Max pages to scrape (default 50): ") or 50)
        
        cars_data = scraper.scrape_multiple_pages(
            max_pages=max_pages,
            delay_range=(2, 5),        # 2-5 seconds between pages
            request_delay=(0.5, 1.5)   # 0.5-1.5 seconds per request
        )
    
    # Convert to DataFrame
    df = pd.DataFrame(cars_data)
    
    # Clean and preprocess
    df = df.dropna(subset=['price', 'mileage'])  # Remove rows with missing key data
    df = scraper.create_binary_variables(df)
    
    # Save to CSV with timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f'used_cars_dataset_{timestamp}.csv'
    df.to_csv(filename, index=False)
    print(f"Scraped {len(df)} cars and saved to {filename}")
    
    # Display basic info
    print("\nDataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    if len(df) > 0:
        print(f"Price range: ${df['price'].min():,} - ${df['price'].max():,}")
        print(f"Year range: {df['year'].min()} - {df['year'].max()}")
        if 'scrape_location' in df.columns:
            print(f"Locations covered: {df['scrape_location'].nunique()}")
            print(f"Top 5 locations by count:")
            print(df['scrape_location'].value_counts().head())

if __name__ == "__main__":
    main()