"""
HORSE RACING DATA GENERATOR
===========================

Generates realistic horse racing data at scale.
Creates 1000+ horses across multiple races, tracks, and time periods.

Save as: data_generator.py
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

class HorseRacingDataGenerator:
    def __init__(self):
        self.horses = self._create_horse_pool()
        self.jockeys = self._create_jockey_pool()
        self.trainers = self._create_trainer_pool()
        self.tracks = self._create_track_pool()
        
    def _create_horse_pool(self):
        """Create a realistic pool of horse names"""
        prefixes = ['Thunder', 'Lightning', 'Storm', 'Fire', 'Golden', 'Silver', 'Midnight', 
                   'Speed', 'Star', 'Wild', 'Desert', 'Ocean', 'Mountain', 'Valley', 'Royal',
                   'Blazing', 'Flying', 'Dancing', 'Running', 'Jumping', 'Galloping', 'Swift',
                   'Bold', 'Brave', 'Noble', 'Proud', 'Strong', 'Fast', 'Quick', 'Rapid']
        
        suffixes = ['Strike', 'Bolt', 'Chaser', 'Wind', 'Arrow', 'Bullet', 'Runner', 'Demon',
                   'Gazer', 'Spirit', 'Storm', 'Breeze', 'Peak', 'Fire', 'Thunder', 'Fury',
                   'Flash', 'Dash', 'Rush', 'Blaze', 'Comet', 'Rocket', 'Jet', 'Streak',
                   'Express', 'Warrior', 'Champion', 'King', 'Queen', 'Prince', 'Knight']
        
        horses = []
        for i in range(200):  # Create 200 unique horse names
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes)
            horses.append(f"{prefix} {suffix}")
        
        return list(set(horses))  # Remove duplicates
    
    def _create_jockey_pool(self):
        """Create realistic jockey names with skill levels"""
        jockeys = [
            {'name': 'J. Rosario', 'skill': 0.85},
            {'name': 'F. Prat', 'skill': 0.82},
            {'name': 'M. Smith', 'skill': 0.80},
            {'name': 'L. Saez', 'skill': 0.78},
            {'name': 'I. Ortiz', 'skill': 0.76},
            {'name': 'J. Castellano', 'skill': 0.75},
            {'name': 'T. Baze', 'skill': 0.72},
            {'name': 'R. Santana', 'skill': 0.70},
            {'name': 'J. Velazquez', 'skill': 0.68},
            {'name': 'K. Desormeaux', 'skill': 0.65},
            {'name': 'M. Garcia', 'skill': 0.62},
            {'name': 'D. Van Dyke', 'skill': 0.60},
            {'name': 'A. Cedillo', 'skill': 0.58},
            {'name': 'E. Maldonado', 'skill': 0.55},
            {'name': 'R. Gonzalez', 'skill': 0.52},
            {'name': 'C. Lopez', 'skill': 0.50},
            {'name': 'M. Franco', 'skill': 0.48},
            {'name': 'J. Lezcano', 'skill': 0.45},
            {'name': 'D. Davis', 'skill': 0.42},
            {'name': 'A. Gryder', 'skill': 0.40}
        ]
        return jockeys
    
    def _create_trainer_pool(self):
        """Create realistic trainer names with skill levels"""
        trainers = [
            {'name': 'B. Baffert', 'skill': 0.88},
            {'name': 'T. Pletcher', 'skill': 0.85},
            {'name': 'J. Sadler', 'skill': 0.82},
            {'name': 'D. O\'Neill', 'skill': 0.80},
            {'name': 'C. Brown', 'skill': 0.78},
            {'name': 'J. Shirreffs', 'skill': 0.75},
            {'name': 'P. Miller', 'skill': 0.72},
            {'name': 'R. Baltas', 'skill': 0.70},
            {'name': 'M. Casse', 'skill': 0.68},
            {'name': 'S. Asmussen', 'skill': 0.65},
            {'name': 'K. McPeek', 'skill': 0.62},
            {'name': 'W. Mott', 'skill': 0.60},
            {'name': 'M. Maker', 'skill': 0.58},
            {'name': 'R. Mandella', 'skill': 0.55},
            {'name': 'J. Hollendorfer', 'skill': 0.52},
            {'name': 'L. Jones', 'skill': 0.50},
            {'name': 'P. Eurton', 'skill': 0.48},
            {'name': 'G. Weaver', 'skill': 0.45},
            {'name': 'H. Bond', 'skill': 0.42},
            {'name': 'A. Dutrow', 'skill': 0.40}
        ]
        return trainers
    
    def _create_track_pool(self):
        """Create realistic track information"""
        tracks = [
            {'name': 'Santa Anita', 'bias': {'dirt_inside': 1.05, 'turf_pace': 1.02}},
            {'name': 'Churchill Downs', 'bias': {'dirt_speed': 1.04, 'turf_outside': 1.03}},
            {'name': 'Belmont Park', 'bias': {'turf_stamina': 1.06, 'dirt_pace': 1.02}},
            {'name': 'Saratoga', 'bias': {'turf_class': 1.05, 'dirt_inside': 1.03}},
            {'name': 'Del Mar', 'bias': {'turf_speed': 1.04, 'dirt_outside': 1.02}},
            {'name': 'Gulfstream Park', 'bias': {'dirt_speed': 1.03, 'turf_pace': 1.04}},
            {'name': 'Keeneland', 'bias': {'turf_stamina': 1.05, 'dirt_class': 1.03}},
            {'name': 'Oaklawn Park', 'bias': {'dirt_inside': 1.04, 'surface_dirt': 1.02}},
            {'name': 'Aqueduct', 'bias': {'dirt_pace': 1.03, 'winter_track': 1.02}},
            {'name': 'Golden Gate Fields', 'bias': {'turf_outside': 1.04, 'synthetic': 1.03}}
        ]
        return tracks
    
    def _calculate_base_time(self, distance, surface):
        """Calculate realistic base times for different distances and surfaces"""
        base_times = {
            # Dirt times (seconds)
            ('5.0f', 'Dirt'): 58.0,
            ('5.5f', 'Dirt'): 64.0,
            ('6.0f', 'Dirt'): 70.0,
            ('6.5f', 'Dirt'): 76.5,
            ('7.0f', 'Dirt'): 83.0,
            ('1m', 'Dirt'): 96.0,
            ('1.125m', 'Dirt'): 108.0,
            ('1.25m', 'Dirt'): 120.0,
            ('1.5m', 'Dirt'): 150.0,
            
            # Turf times (slightly slower)
            ('5.0f', 'Turf'): 59.0,
            ('5.5f', 'Turf'): 65.0,
            ('6.0f', 'Turf'): 71.0,
            ('6.5f', 'Turf'): 77.5,
            ('7.0f', 'Turf'): 84.0,
            ('1m', 'Turf'): 97.0,
            ('1.125m', 'Turf'): 109.0,
            ('1.25m', 'Turf'): 121.0,
            ('1.5m', 'Turf'): 152.0,
        }
        return base_times.get((distance, surface), 90.0)
    
    def _generate_realistic_odds(self, horse_quality, jockey_skill, trainer_skill, post_position, field_size):
        """Generate realistic odds based on horse and jockey quality"""
        # Base odds calculation
        base_quality = (horse_quality * 0.5 + jockey_skill * 0.3 + trainer_skill * 0.2)
        
        # Post position adjustment
        if post_position <= 3:
            post_adjustment = 1.05  # Inside posts get slight odds boost
        elif post_position >= field_size - 2:
            post_adjustment = 0.95  # Outside posts get penalized
        else:
            post_adjustment = 1.0
        
        # Convert quality to odds (inverse relationship)
        adjusted_quality = base_quality * post_adjustment
        
        # Generate odds (favorites have low odds, longshots have high odds)
        if adjusted_quality > 0.8:
            odds = np.random.uniform(1.5, 4.0)  # Favorites
        elif adjusted_quality > 0.6:
            odds = np.random.uniform(3.0, 8.0)  # Second choices
        elif adjusted_quality > 0.4:
            odds = np.random.uniform(6.0, 15.0)  # Medium odds
        else:
            odds = np.random.uniform(12.0, 30.0)  # Longshots
        
        return round(odds, 2)
    
    def _determine_race_outcome(self, horses_in_race):
        """Determine race finishing order based on horse qualities and random factors"""
        # Calculate win probability for each horse
        for horse in horses_in_race:
            base_prob = (horse['horse_quality'] * 0.4 + 
                        horse['jockey_skill'] * 0.3 + 
                        horse['trainer_skill'] * 0.2 +
                        horse['track_advantage'] * 0.1)
            
            # Add random racing luck
            racing_luck = np.random.normal(0, 0.15)
            horse['final_performance'] = base_prob + racing_luck
        
        # Sort by final performance to determine finishing order
        horses_in_race.sort(key=lambda x: x['final_performance'], reverse=True)
        
        # Assign finishing positions
        for i, horse in enumerate(horses_in_race):
            horse['finish_position'] = i + 1
            
            # Calculate margins (seconds behind winner)
            if i == 0:
                horse['margin'] = "Won by " + str(round(np.random.uniform(0.5, 3.0), 1))
            else:
                horse['margin'] = str(round(np.random.uniform(0.25, 2.0 * i), 1))
        
        return horses_in_race
    
    def generate_race_data(self, num_races=125, horses_per_race=8):
        """Generate realistic race data"""
        print(f"Generating {num_races} races with {horses_per_race} horses each...")
        print(f"Total horses: {num_races * horses_per_race}")
        
        all_races = []
        race_date = datetime(2024, 1, 1)
        
        for race_num in range(num_races):
            # Advance date occasionally
            if race_num % 10 == 0:
                race_date += timedelta(days=1)
            
            track = random.choice(self.tracks)
            race_number = (race_num % 10) + 1
            
            # Race characteristics
            distances = ['5.0f', '5.5f', '6.0f', '6.5f', '7.0f', '1m', '1.125m', '1.25m']
            surfaces = ['Dirt', 'Turf']
            conditions = ['Fast', 'Good', 'Firm', 'Yielding', 'Sloppy', 'Muddy']
            
            race_distance = random.choice(distances)
            race_surface = random.choice(surfaces)
            track_condition = random.choice(conditions)
            
            # Generate horses for this race
            race_horses = []
            used_horses = set()
            
            for post_pos in range(1, horses_per_race + 1):
                # Select unique horse for this race
                horse_name = random.choice(self.horses)
                while horse_name in used_horses:
                    horse_name = random.choice(self.horses)
                used_horses.add(horse_name)
                
                # Select jockey and trainer
                jockey = random.choice(self.jockeys)
                trainer = random.choice(self.trainers)
                
                # Horse quality (varies for each race)
                horse_quality = np.random.beta(2, 3)  # Skewed towards lower quality
                
                # Track advantage based on surface/distance preferences
                track_advantage = np.random.normal(0.5, 0.1)
                
                # Generate odds
                odds = self._generate_realistic_odds(
                    horse_quality, jockey['skill'], trainer['skill'], 
                    post_pos, horses_per_race
                )
                
                horse_data = {
                    'race_date': race_date.strftime('%Y-%m-%d'),
                    'track_name': track['name'],
                    'race_number': race_number,
                    'distance': race_distance,
                    'surface': race_surface,
                    'track_condition': track_condition,
                    'horse_name': horse_name,
                    'jockey': jockey['name'],
                    'trainer': trainer['name'],
                    'post_position': post_pos,
                    'odds': odds,
                    'weight': random.randint(115, 126),
                    'horse_quality': horse_quality,
                    'jockey_skill': jockey['skill'],
                    'trainer_skill': trainer['skill'],
                    'track_advantage': track_advantage
                }
                
                race_horses.append(horse_data)
            
            # Determine race outcome
            race_horses = self._determine_race_outcome(race_horses)
            
            # Generate final times based on performance
            base_time = self._calculate_base_time(race_distance, race_surface)
            winner_time = base_time + np.random.normal(0, 2.0)
            
            for horse in race_horses:
                # Time penalty based on finishing position
                time_penalty = (horse['finish_position'] - 1) * 0.2
                final_time = winner_time + time_penalty + np.random.normal(0, 0.5)
                
                # Generate speed rating (inverse of time performance)
                speed_rating = max(40, min(120, int(100 - (final_time - base_time) * 5)))
                
                # Clean up data for output
                race_entry = {
                    'race_date': horse['race_date'],
                    'track_name': horse['track_name'],
                    'race_number': horse['race_number'],
                    'distance': horse['distance'],
                    'surface': horse['surface'],
                    'track_condition': horse['track_condition'],
                    'horse_name': horse['horse_name'],
                    'jockey': horse['jockey'],
                    'trainer': horse['trainer'],
                    'post_position': horse['post_position'],
                    'final_time': f"{int(final_time // 60)}:{final_time % 60:05.2f}",
                    'finish_position': horse['finish_position'],
                    'odds': horse['odds'],
                    'weight': horse['weight'],
                    'margin': horse['margin'],
                    'speed_rating': speed_rating
                }
                
                all_races.append(race_entry)
            
            # Progress indicator
            if (race_num + 1) % 25 == 0:
                print(f"Generated {race_num + 1} races...")
        
        return all_races
    
    def save_data(self, race_data, filename='horse_racing_data.csv'):
        """Save race data to CSV"""
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        
        df = pd.DataFrame(race_data)
        df.to_csv(filepath, index=False)
        
        print(f"\nSUCCESS: Saved {len(race_data)} race entries to {filepath}")
        
        # Show summary statistics
        print(f"\nDATA SUMMARY:")
        print(f"Total horses: {len(race_data)}")
        print(f"Total races: {df['race_number'].nunique() * df['track_name'].nunique()}")
        print(f"Date range: {df['race_date'].min()} to {df['race_date'].max()}")
        print(f"Tracks: {df['track_name'].nunique()}")
        print(f"Unique horses: {df['horse_name'].nunique()}")
        print(f"Unique jockeys: {df['jockey'].nunique()}")
        print(f"Win rate: {(df['finish_position'] == 1).mean():.1%}")
        
        return filepath

def main():
    """Generate 1000+ horses across multiple races"""
    print("HORSE RACING DATA GENERATOR")
    print("=" * 50)
    print("Scaling from 40 to 1000+ horses...")
    
    generator = HorseRacingDataGenerator()
    
    # Generate 125 races with 8 horses each = 1000 horses
    race_data = generator.generate_race_data(num_races=125, horses_per_race=8)
    
    # Save the data
    filepath = generator.save_data(race_data, 'horse_racing_data_1000.csv')
    
    print(f"\nREADY FOR MACHINE LEARNING!")
    print(f"Next step: Run preprocess.py to prepare this data for win prediction")
    
    return filepath

if __name__ == "__main__":
    main()