import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from datetime import datetime
import json

class HorseRacingScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
    def test_connection(self, url):
        """Test if we can connect to a URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            print(f"Status Code: {response.status_code}")
            print(f"URL: {url}")
            if response.status_code == 200:
                return True
            else:
                return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def create_sample_data(self):
        """Create sample horse racing data for testing our ML pipeline"""
        print("Creating sample horse racing data...")
        
        # Sample data that mimics real horse racing results
        sample_races = [
            {
                'race_date': '2024-07-01',
                'track_name': 'Santa Anita',
                'race_number': 1,
                'distance': '6.0f',
                'surface': 'Dirt',
                'track_condition': 'Fast',
                'horse_name': 'Thunder Strike',
                'jockey': 'J. Rosario',
                'trainer': 'B. Baffert',
                'post_position': 3,
                'final_time': '1:10.23',
                'finish_position': 1,
                'odds': '3.20',
                'weight': 118,
                'margin': 'Won by 2',
                'speed_rating': 95
            },
            {
                'race_date': '2024-07-01',
                'track_name': 'Santa Anita',
                'race_number': 1,
                'distance': '6.0f',
                'surface': 'Dirt',
                'track_condition': 'Fast',
                'horse_name': 'Lightning Bolt',
                'jockey': 'F. Prat',
                'trainer': 'J. Sadler',
                'post_position': 5,
                'final_time': '1:10.45',
                'finish_position': 2,
                'odds': '5.60',
                'weight': 118,
                'margin': '2',
                'speed_rating': 92
            },
            {
                'race_date': '2024-07-01',
                'track_name': 'Santa Anita',
                'race_number': 1,
                'distance': '6.0f',
                'surface': 'Dirt',
                'track_condition': 'Fast',
                'horse_name': 'Storm Chaser',
                'jockey': 'M. Smith',
                'trainer': 'D. O\'Neill',
                'post_position': 1,
                'final_time': '1:10.67',
                'finish_position': 3,
                'odds': '8.40',
                'weight': 118,
                'margin': '1.5',
                'speed_rating': 89
            }
        ]
        
        # Generate more sample data for different races/days
        import random
        
        horses = ['Thunder Strike', 'Lightning Bolt', 'Storm Chaser', 'Fire Wind', 'Golden Arrow', 
                 'Silver Bullet', 'Midnight Runner', 'Speed Demon', 'Star Gazer', 'Wild Spirit']
        jockeys = ['J. Rosario', 'F. Prat', 'M. Smith', 'L. Saez', 'I. Ortiz', 'J. Castellano']
        trainers = ['B. Baffert', 'J. Sadler', 'D. O\'Neill', 'C. Brown', 'T. Pletcher', 'J. Shirreffs']
        
        all_races = []
        
        for race_num in range(1, 6):  # 5 races
            for pos in range(1, 9):  # 8 horses per race
                race_data = {
                    'race_date': '2024-07-01',
                    'track_name': 'Santa Anita',
                    'race_number': race_num,
                    'distance': random.choice(['6.0f', '6.5f', '7.0f', '1m', '1.25m']),
                    'surface': random.choice(['Dirt', 'Turf']),
                    'track_condition': random.choice(['Fast', 'Good', 'Sloppy']),
                    'horse_name': random.choice(horses),
                    'jockey': random.choice(jockeys),
                    'trainer': random.choice(trainers),
                    'post_position': random.randint(1, 8),
                    'final_time': f"1:{random.randint(8, 15)}.{random.randint(10, 99)}",
                    'finish_position': pos,
                    'odds': f"{random.uniform(2.0, 20.0):.2f}",
                    'weight': random.randint(115, 126),
                    'margin': f"{random.uniform(0.5, 5.0):.1f}" if pos > 1 else "Won by 1.5",
                    'speed_rating': random.randint(75, 100)
                }
                all_races.append(race_data)
        
        return all_races
    
    def save_to_csv(self, race_data, filename):
        """Save race data to CSV file"""
        if race_data:
            df = pd.DataFrame(race_data)
            
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            filepath = os.path.join('data', filename)
            df.to_csv(filepath, index=False)
            print(f"Saved {len(race_data)} records to {filepath}")
            
            # Show sample of the data
            print("\nSample of saved data:")
            print(df.head())
            
            return filepath
        else:
            print("No data to save")
            return None
    
    def scrape_racing_reference(self, track_code, date):
        """
        Alternative scraper for Racing Reference (often more accessible)
        This is a backup plan if Equibase doesn't work
        """
        # This would be implemented later with actual Racing Reference URLs
        print(f"Racing Reference scraper not implemented yet")
        return []

# Example usage and testing
if __name__ == "__main__":
    scraper = HorseRacingScraper()
    
    print("🏇 Horse Racing Data Scraper")
    print("=" * 50)
    
    # Test different approaches
    print("\n1. Testing Equibase connection...")
    equibase_urls = [
        "https://www.equibase.com",
        "https://www.equibase.com/static/chart/summary/",
        "https://www.equibase.com/static/entry/index.html"
    ]
    
    for url in equibase_urls:
        print(f"Testing: {url}")
        if scraper.test_connection(url):
            print("✅ Connection successful")
        else:
            print("❌ Connection failed")
        print()
    
    print("\n2. Creating sample data for ML development...")
    sample_data = scraper.create_sample_data()
    
    if sample_data:
        filename = "sample_race_data.csv"
        scraper.save_to_csv(sample_data, filename)
        print(f"\n✅ Success! Created {len(sample_data)} sample records")
        print(f"📁 Data saved to: data/{filename}")
        print("\n💡 Next steps:")
        print("   - Use this sample data to build your ML model")
        print("   - Test your preprocessing and training pipeline")
        print("   - Once ML is working, we'll get real data")
    else:
        print("❌ No sample data created")
    
    print("\n" + "=" * 50)
    print("🎯 STRATEGY: Start with sample data, build ML pipeline first!")
    print("   Real data scraping can be tricky, but ML development doesn't need to wait.")