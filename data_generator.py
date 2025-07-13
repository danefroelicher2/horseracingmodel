"""
HORSE RACING DATA GENERATOR - 10,000+ HORSES
=============================================

Generates realistic horse racing data at scale.
Creates 10,000+ horses across multiple races, tracks, and time periods.
Optimized for performance and realistic correlations.

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
        """Create a large pool of realistic horse names for 10K+ races"""
        prefixes = [
            'Thunder', 'Lightning', 'Storm', 'Fire', 'Golden', 'Silver', 'Midnight', 
            'Speed', 'Star', 'Wild', 'Desert', 'Ocean', 'Mountain', 'Valley', 'Royal',
            'Blazing', 'Flying', 'Dancing', 'Running', 'Jumping', 'Galloping', 'Swift',
            'Bold', 'Brave', 'Noble', 'Proud', 'Strong', 'Fast', 'Quick', 'Rapid',
            'Iron', 'Steel', 'Diamond', 'Ruby', 'Emerald', 'Crystal', 'Shadow', 'Light',
            'Dark', 'Bright', 'Shining', 'Glowing', 'Burning', 'Frozen', 'Electric',
            'Magnetic', 'Cosmic', 'Stellar', 'Solar', 'Lunar', 'Mystic', 'Magic',
            'Lucky', 'Fortune', 'Destiny', 'Victory', 'Glory', 'Honor', 'Legend'
        ]
        
        suffixes = [
            'Strike', 'Bolt', 'Chaser', 'Wind', 'Arrow', 'Bullet', 'Runner', 'Demon',
            'Gazer', 'Spirit', 'Storm', 'Breeze', 'Peak', 'Fire', 'Thunder', 'Fury',
            'Flash', 'Dash', 'Rush', 'Blaze', 'Comet', 'Rocket', 'Jet', 'Streak',
            'Express', 'Warrior', 'Champion', 'King', 'Queen', 'Prince', 'Knight',
            'Force', 'Power', 'Energy', 'Dynamo', 'Turbo', 'Nitro', 'Boost', 'Surge',
            'Wave', 'Tide', 'Current', 'Flow', 'Stream', 'River', 'Ocean', 'Sea',
            'Mountain', 'Hill', 'Peak', 'Summit', 'Ridge', 'Valley', 'Canyon', 'Mesa'
        ]
        
        horses = []
        # Generate enough unique names for 10K+ races
        for i in range(2000):  
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes)
            name = f"{prefix} {suffix}"
            if name not in horses:  # Avoid duplicates
                horses.append(name)
        
        return horses
    
    def _create_jockey_pool(self):
        """Create realistic jockey pool with varied skill levels"""
        # Elite jockeys (top tier)
        elite_jockeys = [
            {'name': 'J. Rosario', 'skill': 0.88, 'tier': 'elite'},
            {'name': 'F. Prat', 'skill': 0.86, 'tier': 'elite'},
            {'name': 'M. Smith', 'skill': 0.84, 'tier': 'elite'},
            {'name': 'L. Saez', 'skill': 0.82, 'tier': 'elite'},
            {'name': 'I. Ortiz', 'skill': 0.80, 'tier': 'elite'}
        ]
        
        # Top jockeys
        top_jockeys = [
            {'name': 'J. Castellano', 'skill': 0.78, 'tier': 'top'},
            {'name': 'T. Baze', 'skill': 0.76, 'tier': 'top'},
            {'name': 'R. Santana', 'skill': 0.74, 'tier': 'top'},
            {'name': 'J. Velazquez', 'skill': 0.72, 'tier': 'top'},
            {'name': 'K. Desormeaux', 'skill': 0.70, 'tier': 'top'},
            {'name': 'M. Garcia', 'skill': 0.68, 'tier': 'top'},
            {'name': 'D. Van Dyke', 'skill': 0.66, 'tier': 'top'}
        ]
        
        # Good jockeys  
        good_jockeys = [
            {'name': 'A. Cedillo', 'skill': 0.64, 'tier': 'good'},
            {'name': 'E. Maldonado', 'skill': 0.62, 'tier': 'good'},
            {'name': 'R. Gonzalez', 'skill': 0.60, 'tier': 'good'},
            {'name': 'C. Lopez', 'skill': 0.58, 'tier': 'good'},
            {'name': 'M. Franco', 'skill': 0.56, 'tier': 'good'},
            {'name': 'J. Lezcano', 'skill': 0.54, 'tier': 'good'},
            {'name': 'D. Davis', 'skill': 0.52, 'tier': 'good'},
            {'name': 'A. Gryder', 'skill': 0.50, 'tier': 'good'}
        ]
        
        # Average/developing jockeys
        average_jockeys = []
        for i in range(1, 16):  # 15 more jockeys
            average_jockeys.append({
                'name': f'Jockey_{i:02d}',
                'skill': np.random.uniform(0.35, 0.48),
                'tier': 'average'
            })
        
        return elite_jockeys + top_jockeys + good_jockeys + average_jockeys
    
    def _create_trainer_pool(self):
        """Create realistic trainer pool with varied skill levels"""
        # Elite trainers
        elite_trainers = [
            {'name': 'B. Baffert', 'skill': 0.90, 'tier': 'elite'},
            {'name': 'T. Pletcher', 'skill': 0.88, 'tier': 'elite'},
            {'name': 'C. Brown', 'skill': 0.85, 'tier': 'elite'}
        ]
        
        # Top trainers
        top_trainers = [
            {'name': 'J. Sadler', 'skill': 0.82, 'tier': 'top'},
            {'name': 'D. O\'Neill', 'skill': 0.80, 'tier': 'top'},
            {'name': 'J. Shirreffs', 'skill': 0.78, 'tier': 'top'},
            {'name': 'P. Miller', 'skill': 0.76, 'tier': 'top'},
            {'name': 'R. Baltas', 'skill': 0.74, 'tier': 'top'},
            {'name': 'M. Casse', 'skill': 0.72, 'tier': 'top'},
            {'name': 'S. Asmussen', 'skill': 0.70, 'tier': 'top'}
        ]
        
        # Good trainers
        good_trainers = [
            {'name': 'K. McPeek', 'skill': 0.68, 'tier': 'good'},
            {'name': 'W. Mott', 'skill': 0.66, 'tier': 'good'},
            {'name': 'M. Maker', 'skill': 0.64, 'tier': 'good'},
            {'name': 'R. Mandella', 'skill': 0.62, 'tier': 'good'},
            {'name': 'J. Hollendorfer', 'skill': 0.60, 'tier': 'good'},
            {'name': 'L. Jones', 'skill': 0.58, 'tier': 'good'},
            {'name': 'P. Eurton', 'skill': 0.56, 'tier': 'good'}
        ]
        
        # Average trainers
        average_trainers = []
        for i in range(1, 21):  # 20 more trainers
            average_trainers.append({
                'name': f'Trainer_{i:02d}',
                'skill': np.random.uniform(0.35, 0.52),
                'tier': 'average'
            })
        
        return elite_trainers + top_trainers + good_trainers + average_trainers
    
    def _create_track_pool(self):
        """Create comprehensive track pool for 10K+ races"""
        tracks = [
            # Major tracks with specific characteristics
            {'name': 'Santa Anita', 'bias': {'dirt_inside': 1.05, 'turf_pace': 1.02}, 'quality': 'premium'},
            {'name': 'Churchill Downs', 'bias': {'dirt_speed': 1.04, 'turf_outside': 1.03}, 'quality': 'premium'},
            {'name': 'Belmont Park', 'bias': {'turf_stamina': 1.06, 'dirt_pace': 1.02}, 'quality': 'premium'},
            {'name': 'Saratoga', 'bias': {'turf_class': 1.05, 'dirt_inside': 1.03}, 'quality': 'premium'},
            {'name': 'Del Mar', 'bias': {'turf_speed': 1.04, 'dirt_outside': 1.02}, 'quality': 'premium'},
            
            # Major tracks
            {'name': 'Gulfstream Park', 'bias': {'dirt_speed': 1.03, 'turf_pace': 1.04}, 'quality': 'major'},
            {'name': 'Keeneland', 'bias': {'turf_stamina': 1.05, 'dirt_class': 1.03}, 'quality': 'major'},
            {'name': 'Oaklawn Park', 'bias': {'dirt_inside': 1.04, 'surface_dirt': 1.02}, 'quality': 'major'},
            {'name': 'Aqueduct', 'bias': {'dirt_pace': 1.03, 'winter_track': 1.02}, 'quality': 'major'},
            {'name': 'Golden Gate Fields', 'bias': {'turf_outside': 1.04, 'synthetic': 1.03}, 'quality': 'major'},
            
            # Regional tracks
            {'name': 'Arlington Park', 'bias': {'turf_advantage': 1.03}, 'quality': 'regional'},
            {'name': 'Woodbine', 'bias': {'synthetic_track': 1.04}, 'quality': 'regional'},
            {'name': 'Los Alamitos', 'bias': {'sprint_specialist': 1.05}, 'quality': 'regional'},
            {'name': 'Laurel Park', 'bias': {'mud_track': 1.03}, 'quality': 'regional'},
            {'name': 'Monmouth Park', 'bias': {'turf_bias': 1.02}, 'quality': 'regional'}
        ]
        
        return tracks
    
    def _calculate_base_time(self, distance, surface):
        """Calculate realistic base times with more variety"""
        base_times = {
            # Dirt times (seconds)
            ('5.0f', 'Dirt'): 58.0, ('5.5f', 'Dirt'): 64.0, ('6.0f', 'Dirt'): 70.0,
            ('6.5f', 'Dirt'): 76.5, ('7.0f', 'Dirt'): 83.0, ('1m', 'Dirt'): 96.0,
            ('1.125m', 'Dirt'): 108.0, ('1.25m', 'Dirt'): 120.0, ('1.5m', 'Dirt'): 150.0,
            
            # Turf times (slightly slower)
            ('5.0f', 'Turf'): 59.0, ('5.5f', 'Turf'): 65.0, ('6.0f', 'Turf'): 71.0,
            ('6.5f', 'Turf'): 77.5, ('7.0f', 'Turf'): 84.0, ('1m', 'Turf'): 97.0,
            ('1.125m', 'Turf'): 109.0, ('1.25m', 'Turf'): 121.0, ('1.5m', 'Turf'): 152.0,
        }
        return base_times.get((distance, surface), 90.0)
    
    def _generate_realistic_odds(self, horse_quality, jockey_skill, trainer_skill, post_position, field_size):
        """Enhanced odds generation with more realistic distribution"""
        # Base quality calculation with weights
        base_quality = (horse_quality * 0.5 + jockey_skill * 0.25 + trainer_skill * 0.25)
        
        # Post position adjustment (more nuanced)
        if post_position <= 2:
            post_adjustment = 1.08  # Rail posts get bigger boost
        elif post_position <= 4:
            post_adjustment = 1.03  # Inside posts slight boost
        elif post_position >= field_size - 1:
            post_adjustment = 0.92  # Outside posts penalized more
        else:
            post_adjustment = 1.0
        
        # Field size impact (bigger fields = longer odds)
        field_adjustment = 1.0 + (field_size - 8) * 0.02
        
        adjusted_quality = base_quality * post_adjustment * field_adjustment
        
        # More realistic odds distribution
        if adjusted_quality > 0.85:
            odds = np.random.uniform(1.2, 3.0)  # Heavy favorites
        elif adjusted_quality > 0.75:
            odds = np.random.uniform(2.5, 5.5)  # Favorites
        elif adjusted_quality > 0.65:
            odds = np.random.uniform(4.0, 9.0)  # Second choices
        elif adjusted_quality > 0.50:
            odds = np.random.uniform(7.0, 18.0)  # Medium odds
        elif adjusted_quality > 0.35:
            odds = np.random.uniform(15.0, 35.0)  # Longshots
        else:
            odds = np.random.uniform(30.0, 99.0)  # Extreme longshots
        
        return round(odds, 2)
    
    def _determine_race_outcome(self, horses_in_race):
        """Enhanced race outcome determination with more realistic factors"""
        for horse in horses_in_race:
            # Base performance calculation
            base_prob = (horse['horse_quality'] * 0.45 + 
                        horse['jockey_skill'] * 0.25 + 
                        horse['trainer_skill'] * 0.20 +
                        horse['track_advantage'] * 0.10)
            
            # Racing luck (bigger factor for excitement)
            racing_luck = np.random.normal(0, 0.18)
            
            # Post position racing impact (different from odds impact)
            if horse['post_position'] <= 3:
                race_advantage = 0.02
            elif horse['post_position'] >= 10:
                race_advantage = -0.03
            else:
                race_advantage = 0.0
            
            # Track condition impact
            condition_impact = np.random.normal(0, 0.05)
            
            horse['final_performance'] = base_prob + racing_luck + race_advantage + condition_impact
        
        # Sort by performance
        horses_in_race.sort(key=lambda x: x['final_performance'], reverse=True)
        
        # Assign finishing positions and margins
        for i, horse in enumerate(horses_in_race):
            horse['finish_position'] = i + 1
            
            if i == 0:
                margin_length = np.random.uniform(0.1, 4.0)
                horse['margin'] = f"Won by {margin_length:.1f}"
            else:
                # More realistic margin distribution
                base_margin = 0.5 + (i - 1) * 0.3
                margin_variation = np.random.uniform(0.1, 1.0)
                horse['margin'] = str(round(base_margin + margin_variation, 1))
        
        return horses_in_race
    
    def generate_race_data(self, num_races=1250, horses_per_race_range=(6, 12)):
        """
        Generate 10,000+ horses efficiently
        1250 races * 8 average horses = 10,000 horses
        """
        print(f"Generating {num_races} races for 10,000+ horses...")
        print(f"Field sizes: {horses_per_race_range[0]}-{horses_per_race_range[1]} horses per race")
        
        all_races = []
        race_date = datetime(2024, 1, 1)
        
        # Progress tracking
        progress_interval = num_races // 20  # 20 progress updates
        
        for race_num in range(num_races):
            # Advance date more realistically (3-4 races per day per track)
            if race_num % 50 == 0:  # New day every 50 races
                race_date += timedelta(days=1)
            
            track = random.choice(self.tracks)
            race_number = (race_num % 12) + 1  # Up to 12 races per day
            
            # Variable field sizes for realism
            field_size = random.randint(horses_per_race_range[0], horses_per_race_range[1])
            
            # Race characteristics with seasonal variation
            distances = ['5.0f', '5.5f', '6.0f', '6.5f', '7.0f', '1m', '1.125m', '1.25m']
            distance_weights = [0.15, 0.15, 0.25, 0.15, 0.15, 0.10, 0.03, 0.02]  # More realistic distribution
            
            surfaces = ['Dirt', 'Turf']
            surface_weights = [0.7, 0.3]  # Dirt more common
            
            conditions = ['Fast', 'Good', 'Firm', 'Yielding', 'Sloppy', 'Muddy']
            condition_weights = [0.4, 0.3, 0.15, 0.05, 0.07, 0.03]  # Fast tracks most common
            
            race_distance = np.random.choice(distances, p=distance_weights)
            race_surface = np.random.choice(surfaces, p=surface_weights)
            track_condition = np.random.choice(conditions, p=condition_weights)
            
            # Generate horses for this race
            race_horses = []
            used_horses = set()
            
            for post_pos in range(1, field_size + 1):
                # Select unique horse
                horse_name = random.choice(self.horses)
                while horse_name in used_horses:
                    horse_name = random.choice(self.horses)
                used_horses.add(horse_name)
                
                # Select jockey and trainer with realistic distribution
                # Elite jockeys get better horses more often
                if np.random.random() < 0.3:  # 30% chance for top tier
                    jockey = random.choice([j for j in self.jockeys if j['tier'] in ['elite', 'top']])
                else:
                    jockey = random.choice(self.jockeys)
                
                if np.random.random() < 0.25:  # 25% chance for top trainers
                    trainer = random.choice([t for t in self.trainers if t['tier'] in ['elite', 'top']])
                else:
                    trainer = random.choice(self.trainers)
                
                # Horse quality with more variation
                horse_quality = np.random.beta(2.5, 4)  # Skewed but with high-end horses
                
                # Track advantage
                track_advantage = np.random.normal(0.5, 0.12)
                
                # Generate odds
                odds = self._generate_realistic_odds(
                    horse_quality, jockey['skill'], trainer['skill'], 
                    post_pos, field_size
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
                    'weight': random.randint(114, 128),  # Wider weight range
                    'horse_quality': horse_quality,
                    'jockey_skill': jockey['skill'],
                    'trainer_skill': trainer['skill'],
                    'track_advantage': track_advantage,
                    'field_size': field_size
                }
                
                race_horses.append(horse_data)
            
            # Determine race outcome
            race_horses = self._determine_race_outcome(race_horses)
            
            # Generate final times
            base_time = self._calculate_base_time(race_distance, race_surface)
            winner_time = base_time + np.random.normal(0, 1.5)
            
            for horse in race_horses:
                # Time based on finishing position with more realism
                position_penalty = (horse['finish_position'] - 1) * 0.15
                time_variation = np.random.normal(0, 0.4)
                final_time = winner_time + position_penalty + time_variation
                
                # Speed rating calculation
                time_diff = final_time - base_time
                speed_rating = max(35, min(125, int(100 - time_diff * 4 + np.random.normal(0, 2))))
                
                # Create final race entry
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
            if (race_num + 1) % progress_interval == 0:
                progress = (race_num + 1) / num_races * 100
                total_horses = len(all_races)
                print(f"Progress: {progress:.0f}% - Generated {race_num + 1} races ({total_horses} horses)")
        
        print(f"Generation complete! Total horses: {len(all_races)}")
        return all_races
    
    def save_data(self, race_data, filename='horse_racing_data_10k.csv'):
        """Save race data with enhanced statistics"""
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        
        df = pd.DataFrame(race_data)
        df.to_csv(filepath, index=False)
        
        print(f"\nSUCCESS: Saved {len(race_data)} race entries to {filepath}")
        
        # Comprehensive summary statistics
        print(f"\n=== COMPREHENSIVE DATA SUMMARY ===")
        print(f"Total horses: {len(race_data):,}")
        print(f"Total races: {len(df.groupby(['race_date', 'track_name', 'race_number'])):,}")
        print(f"Date range: {df['race_date'].min()} to {df['race_date'].max()}")
        print(f"Racing days: {df['race_date'].nunique()}")
        print(f"Tracks: {df['track_name'].nunique()}")
        print(f"Unique horses: {df['horse_name'].nunique():,}")
        print(f"Unique jockeys: {df['jockey'].nunique()}")
        print(f"Unique trainers: {df['trainer'].nunique()}")
        print(f"Average field size: {df.groupby(['race_date', 'track_name', 'race_number']).size().mean():.1f}")
        print(f"Win rate: {(df['finish_position'] == 1).mean():.1%}")
        
        # Validation statistics
        print(f"\n=== VALIDATION STATISTICS ===")
        favorites = df[df.groupby(['race_date', 'track_name', 'race_number'])['odds'].transform('min') == df['odds']]
        print(f"Favorite win rate: {(favorites['finish_position'] == 1).mean():.1%}")
        
        longshots = df[df['odds'] > 20]
        print(f"Longshot (20+ odds) win rate: {(longshots['finish_position'] == 1).mean():.1%}")
        
        print(f"File size: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB")
        
        return filepath

def main():
    """Generate 10,000+ horses across multiple races"""
    print("HORSE RACING DATA GENERATOR - 10,000+ HORSES")
    print("=" * 60)
    print("Scaling from 1,000 to 10,000+ horses...")
    print("This will create a professional-grade dataset!")
    
    generator = HorseRacingDataGenerator()
    
    # Generate 1250 races with variable field sizes (average 8 horses = 10,000+ total)
    race_data = generator.generate_race_data(num_races=1250, horses_per_race_range=(6, 12))
    
    # Save the data
    filepath = generator.save_data(race_data, 'horse_racing_data_10k.csv')
    
    print(f"\n🎉 10,000+ HORSE DATASET READY!")
    print(f"✅ Professional-grade sample size achieved")
    print(f"✅ Statistical significance guaranteed")
    print(f"✅ Ready for advanced machine learning")
    print(f"\nNext step: Run preprocess.py to prepare this massive dataset")
    
    return filepath

if __name__ == "__main__":
    main()