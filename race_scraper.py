import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from datetime import datetime

class EquibaseScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.base_url = "https://www.equibase.com"
        
    def scrape_race_results(self, track_code, date, race_number):
        """
        Scrape a single race result from Equibase
        
        Args:
            track_code (str): Track code like 'SA' for Santa Anita
            date (str): Date in MMDDYY format
            race_number (int): Race number
        """
        # Construct URL - this is the Equibase results page format
        url = f"{self.base_url}/static/entry/runningline/runningline.cfm?track={track_code}&raceDate={date}&raceNumber={race_number}"
        
        try:
            print(f"Scraping: {track_code} - {date} - Race {race_number}")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse the race results table
            race_data = []
            
            # Find the results table (this may need adjustment based on actual HTML structure)
            table = soup.find('table', class_='results-table') or soup.find('table')
            
            if table:
                rows = table.find_all('tr')[1:]  # Skip header row
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 8:  # Ensure we have enough columns
                        try:
                            horse_data = {
                                'race_date': date,
                                'track_name': track_code,
                                'race_number': race_number,
                                'finish_position': cols[0].get_text(strip=True),
                                'horse_name': cols[1].get_text(strip=True),
                                'jockey': cols[2].get_text(strip=True),
                                'odds': cols[3].get_text(strip=True),
                                'final_time': cols[4].get_text(strip=True),
                                'margin': cols[5].get_text(strip=True) if len(cols) > 5 else '',
                                'weight': cols[6].get_text(strip=True) if len(cols) > 6 else '',
                                'post_position': cols[7].get_text(strip=True) if len(cols) > 7 else ''
                            }
                            race_data.append(horse_data)
                        except Exception as e:
                            print(f"Error parsing row: {e}")
                            continue
            
            return race_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return []
        except Exception as e:
            print(f"Error parsing data: {e}")
            return []
    
    def scrape_multiple_races(self, track_code, date, num_races=10):
        """
        Scrape multiple races from a single day
        """
        all_races = []
        
        for race_num in range(1, num_races + 1):
            race_data = self.scrape_race_results(track_code, date, race_num)
            all_races.extend(race_data)
            time.sleep(1)  # Be respectful to the server
        
        return all_races
    
    def save_to_csv(self, race_data, filename):
        """
        Save race data to CSV file
        """
        if race_data:
            df = pd.DataFrame(race_data)
            
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            filepath = os.path.join('data', filename)
            df.to_csv(filepath, index=False)
            print(f"Saved {len(race_data)} records to {filepath}")
            return filepath
        else:
            print("No data to save")
            return None

# Example usage and testing
if __name__ == "__main__":
    scraper = EquibaseScraper()
    
    # Test with Santa Anita results
    # Format: MMDDYY (e.g., 070124 for July 1, 2024)
    test_date = "070124"  # July 1, 2024
    test_track = "SA"     # Santa Anita
    
    print("Starting horse racing data scraper...")
    print("=" * 50)
    
    # Scrape a few races as a test
    race_data = scraper.scrape_multiple_races(test_track, test_date, num_races=3)
    
    if race_data:
        filename = f"{test_track}_{test_date}.csv"
        scraper.save_to_csv(race_data, filename)
        print(f"\nSuccess! Scraped {len(race_data)} horse records")
        print(f"Data saved to: data/{filename}")
    else:
        print("No data scraped - check if the date/track combination has results")
        print("You may need to adjust the HTML parsing based on Equibase's current structure")