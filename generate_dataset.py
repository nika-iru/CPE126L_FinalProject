import pandas as pd
import numpy as np
import random
from collections import Counter

def analyze_survey_data(survey_file):

    # Read the survey CSV
    df = pd.read_csv(survey_file)

    # Extract relevant columns and clean
    df_clean = df[[
        'Approximate distance traveled (in kilometers, e.g. 2.5 or 5.0) ',
        'Fare paid (in Philippine Pesos) ',
        'Passenger Type',
        'Time of day you took the ride',
        'Road Condition along the route',
        'Weather during the Ride',
        'Origin Barangay (Pick-up point, e.g. Buhangin, Mintal, Toril) ',
        'Destination Barangay (Drop-off point, e.g. Poblacion, Bangkal, Agdao)'
    ]].copy()

    # Rename columns for easier processing
    df_clean.columns = ['Distance', 'Fare', 'Passenger_Type', 'Time', 'Road', 'Weather', 'Origin', 'Destination']

    # Clean data types
    df_clean['Distance'] = pd.to_numeric(df_clean['Distance'], errors='coerce')
    df_clean['Fare'] = pd.to_numeric(df_clean['Fare'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Distance', 'Fare'])

    # Calculate fare per km for analysis
    df_clean['Fare_Per_Km'] = df_clean['Fare'] / df_clean['Distance']

    return df_clean

def map_survey_categories(df):
    """Map survey categories to standardized format"""

    # Map Passenger Type
    passenger_map = {
        'Student': 'Student',
        'Regular Commuter': 'Regular',
        'Senior Citizen': 'Senior',
        'PWD': 'PWD'
    }
    df['Passenger_Type'] = df['Passenger_Type'].map(passenger_map).fillna('Regular')

    # Map Time of Day
    time_map = {
        'Morning (5 am - 10 am)': 'Morning Peak',
        'Midday (10 am - 2 pm)': 'Midday',
        'Afternoon (2 PM – 6 PM)': 'Afternoon Peak',
        'Evening (6 PM – 10 PM)': 'Evening',
        'Night (10 PM – 5 AM)': 'Evening'
    }
    df['Time'] = df['Time'].map(time_map).fillna('Midday')

    # Map Road Condition
    road_map = {
        'Mostly paved/concrete roads': 'Paved',
        'Mostly unpaved or rough roads': 'Unpaved',
        'Mixed (both paved and unpaved)': 'Unpaved',
        'Difficult terrain (steep, rocky, etc.)': 'Difficult'
    }
    df['Road'] = df['Road'].map(road_map).fillna('Paved')

    # Map Weather
    weather_map = {
        'Clear/sunny': 'Clear',
        'Overcast': 'Overcast',
        'Light rain': 'Rainy',
        'Heavy rain': 'Rainy',
        'Stormy': 'Rainy'
    }
    df['Weather'] = df['Weather'].map(weather_map).fillna('Clear')

    return df

def generate_dataset_from_survey(survey_file, n_samples=500, output_file='habal_habal_dataset.csv'):
    """
    Generate a dataset based on actual survey patterns

    Parameters:
    -----------
    survey_file : str
        Path to the survey CSV file
    n_samples : int
        Number of samples to generate (default: 500)
    output_file : str
        Output filename for the generated dataset
    """

    # Analyze survey data
    survey_df = analyze_survey_data(survey_file)
    survey_df = map_survey_categories(survey_df)

    # Extract distributions from survey

    # Distance distribution parameters
    dist_mean = survey_df['Distance'].mean()
    dist_std = survey_df['Distance'].std()
    dist_min = survey_df['Distance'].min()
    dist_max = survey_df['Distance'].max()

    # Fare per km statistics (for realistic pricing)
    fare_per_km_mean = survey_df['Fare_Per_Km'].mean()
    fare_per_km_std = survey_df['Fare_Per_Km'].std()

    # Categorical distributions
    passenger_dist = survey_df['Passenger_Type'].value_counts(normalize=True).to_dict()
    time_dist = survey_df['Time'].value_counts(normalize=True).to_dict()
    road_dist = survey_df['Road'].value_counts(normalize=True).to_dict()
    weather_dist = survey_df['Weather'].value_counts(normalize=True).to_dict()

    # Get unique barangays
    origins = survey_df['Origin'].dropna().unique().tolist()
    destinations = survey_df['Destination'].dropna().unique().tolist()
    all_barangays = list(set(origins + destinations))

    # Generate synthetic data
    print(f"\n🔧 Generating {n_samples} synthetic samples...")

    np.random.seed(42)
    random.seed(42)

    generated_data = []

    for i in range(n_samples):
        # Generate distance with realistic distribution
        distance = np.random.normal(dist_mean, dist_std)
        distance = max(dist_min, min(dist_max, distance))  # Clamp to observed range
        distance = round(distance, 1)

        # Select categorical features based on survey distribution
        passenger_type = np.random.choice(
            list(passenger_dist.keys()),
            p=list(passenger_dist.values())
        )

        time_of_day = np.random.choice(
            list(time_dist.keys()),
            p=list(time_dist.values())
        )

        road_condition = np.random.choice(
            list(road_dist.keys()),
            p=list(road_dist.values())
        )

        weather = np.random.choice(
            list(weather_dist.keys()),
            p=list(weather_dist.values())
        )

        # Calculate fare based on survey patterns
        # Base fare calculation using survey average rate per km
        base_fare_per_km = np.random.normal(fare_per_km_mean, fare_per_km_std)
        base_fare_per_km = max(10, base_fare_per_km)  # Minimum rate

        fare = distance * base_fare_per_km

        # Add base fare component (observed in survey: even short trips have minimum fare)
        base_fare = 15
        fare += base_fare

        # Apply passenger type discount (based on common practices)
        if passenger_type == 'Student':
            fare *= 0.90  # 10% discount
        elif passenger_type in ['Senior', 'PWD']:
            fare *= 0.85  # 15% discount

        # Apply time of day surcharge
        if time_of_day == 'Morning Peak':
            fare *= 1.10  # +10%
        elif time_of_day == 'Afternoon Peak':
            fare *= 1.15  # +15%

        # Apply road condition surcharge
        if road_condition == 'Unpaved':
            fare *= 1.10  # +10%
        elif road_condition == 'Difficult':
            fare *= 1.25  # +25%

        # Apply weather surcharge
        if weather == 'Overcast':
            fare *= 1.05  # +5%
        elif weather == 'Rainy':
            fare *= 1.20  # +20%

        # Add realistic variance
        fare += np.random.normal(0, 5)

        # Round to whole number (most survey fares are whole numbers)
        fare = max(15, round(fare))

        # Create record
        record = {
            'Distance': distance,
            'Passenger_Type': passenger_type,
            'Time_of_Day': time_of_day,
            'Road_Condition': road_condition,
            'Fare_Charged': fare,
            'Weather_Condition': weather
        }

        generated_data.append(record)

    # Create DataFrame
    df_generated = pd.DataFrame(generated_data)

    # Add the original survey data (mapped to standard format)
    survey_records = []
    for _, row in survey_df.iterrows():
        record = {
            'Distance': row['Distance'],
            'Passenger_Type': row['Passenger_Type'],
            'Time_of_Day': row['Time'],
            'Road_Condition': row['Road'],
            'Fare_Charged': row['Fare'],
            'Weather_Condition': row['Weather']
        }
        survey_records.append(record)

    df_survey = pd.DataFrame(survey_records)

    # Combine survey data with generated data
    df_final = pd.concat([df_survey, df_generated], ignore_index=True)

    # Shuffle the data
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    df_final.to_csv(output_file, index=False)

    return df_final


if __name__ == "__main__":
    # Path to your survey CSV file
    survey_file = 'HABAL-HABAL FARE EXPERIENCE SURVEY (Responses) - Form Responses 1.csv'

    # Generate 500-entry dataset
    df = generate_dataset_from_survey(
        survey_file=survey_file,
        n_samples=500,  # Will generate ~520 total (500 synthetic + ~20 survey responses)
        output_file='habal_habal_dataset.csv'
    )
