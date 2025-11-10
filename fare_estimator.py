import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd


class HabalHabalFareEstimator:
    """
    AI-Powered Fare Estimator using K-Nearest Neighbors algorithm
    for habal-habal rides in Davao City
    """

    def __init__(self, k=5, dataset_path=None):
        self.k = k
        self.model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        self.scaler = StandardScaler()
        self.is_trained = False

        # Store dataset statistics for fare breakdown
        self.dataset_stats = {
            'base_fare': 30,
            'per_km_rate': 12,
            'passenger_discounts': {'Regular': 0, 'Student': 10, 'Senior': 15, 'PWD': 15},
            'time_surcharges': {'Morning Peak': 15, 'Midday': 0, 'Afternoon Peak': 20, 'Evening': 0},
            'road_surcharges': {'Paved': 0, 'Unpaved': 10, 'Difficult': 30},
            'weather_surcharges': {'Clear': 0, 'Overcast': 5, 'Rainy': 25}
        }

        # Feature encoding mappings
        self.passenger_type_map = {
            'Regular': 0,
            'Student': 1,
            'Senior': 2,
            'PWD': 3
        }

        self.time_of_day_map = {
            'Morning Peak': 0,
            'Midday': 1,
            'Afternoon Peak': 2,
            'Evening': 3
        }

        self.road_condition_map = {
            'Paved': 0,
            'Unpaved': 1,
            'Difficult': 2
        }

        self.weather_map = {
            'Clear': 0,
            'Overcast': 1,
            'Rainy': 2
        }

        # Load dataset or generate synthetic data
        if dataset_path:
            self._load_dataset(dataset_path)
        else:
            # Try to load the generated dataset first
            try:
                self._load_dataset('habal_habal_dataset.csv')
            except:
                # Fallback to synthetic data if generated dataset doesn't exist
                self._generate_training_data()

    def _generate_training_data(self):
        """Generate synthetic habal-habal fare data for Davao City"""
        np.random.seed(42)
        n_samples = 800

        # Generate features
        distances = np.random.uniform(0.5, 15.0, n_samples)
        passenger_types = np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.25, 0.15, 0.1])
        time_of_day = np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.2, 0.35, 0.15])
        road_conditions = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
        weather = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.25, 0.15])

        # Calculate base fares with realistic pricing logic
        base_fare = 30  # Base fare in PHP
        per_km_rate = 12  # Per kilometer rate

        fares = base_fare + (distances * per_km_rate)

        # Apply modifiers
        # Passenger type discounts
        passenger_discount = np.where(passenger_types == 1, 0.9,  # Student: 10% off
                                      np.where(passenger_types == 2, 0.85,  # Senior: 15% off
                                               np.where(passenger_types == 3, 0.85, 1.0)))  # PWD: 15% off
        fares *= passenger_discount

        # Time of day surcharges
        time_surcharge = np.where(time_of_day == 0, 1.15,  # Morning peak: +15%
                                  np.where(time_of_day == 2, 1.2, 1.0))  # Afternoon peak: +20%
        fares *= time_surcharge

        # Road condition surcharges
        road_surcharge = np.where(road_conditions == 1, 1.1,  # Unpaved: +10%
                                  np.where(road_conditions == 2, 1.3, 1.0))  # Difficult: +30%
        fares *= road_surcharge

        # Weather surcharges
        weather_surcharge = np.where(weather == 1, 1.05,  # Overcast: +5%
                                     np.where(weather == 2, 1.25, 1.0))  # Rainy: +25%
        fares *= weather_surcharge

        # Add some realistic variance
        fares += np.random.normal(0, 3, n_samples)
        fares = np.clip(fares, 25, 300)  # Reasonable fare range

        # Create feature matrix
        X = np.column_stack([
            distances,
            passenger_types,
            time_of_day,
            road_conditions,
            weather,
            distances * road_conditions,  # Interaction feature
            distances * time_of_day  # Interaction feature
        ])

        # Scale features and train model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, fares)
        self.is_trained = True

        print(f"Model trained with {n_samples} samples")
        print(f"Using KNN with k={self.k} neighbors")

    def _calculate_synthetic_statistics(self, distances, fares):
        """Calculate statistics from synthetic data"""
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()
        lr.fit(distances.reshape(-1, 1), fares)

        self.dataset_stats['base_fare'] = round(max(0, lr.intercept_), 2)
        self.dataset_stats['per_km_rate'] = round(lr.coef_[0], 2)

        print(f"\n📊 Synthetic Data Pricing:")
        print(f"  Base fare: ₱{self.dataset_stats['base_fare']:.2f}")
        print(f"  Per km rate: ₱{self.dataset_stats['per_km_rate']:.2f}/km")

    def _load_dataset(self, csv_path):
        """
        Load dataset from CSV file matching the structure in the progress report

        Expected CSV columns:
        - Distance: float (kilometers)
        - Passenger_Type: str (Regular/Student/Senior/PWD)
        - Time_of_Day: str (Morning Peak/Midday/Afternoon Peak/Evening) OR HH:MM format
        - Road_Condition: str (Paved/Unpaved/Difficult)
        - Weather_Condition: str (Clear/Overcast/Rainy)
        - Fare_Charged: float (PHP)
        """
        try:
            print(f"Loading dataset from: {csv_path}")
            df = pd.read_csv(csv_path)

            # Validate required columns
            required_cols = ['Distance', 'Passenger_Type', 'Time_of_Day',
                             'Road_Condition', 'Weather_Condition', 'Fare_Charged']

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols}. Using column name variations...")
                # Try alternative column names
                col_mapping = {
                    'Distance': ['distance', 'Distance (km)', 'distance_km'],
                    'Passenger_Type': ['passenger_type', 'PassengerType', 'Passenger Type'],
                    'Time_of_Day': ['time_of_day', 'TimeOfDay', 'Time of Day', 'Time'],
                    'Road_Condition': ['road_condition', 'RoadCondition', 'Road Condition'],
                    'Weather_Condition': ['weather_condition', 'WeatherCondition', 'Weather Condition', 'Weather'],
                    'Fare_Charged': ['fare_charged', 'FareCharged', 'Fare Charged', 'Fare', 'fare']
                }

                # Rename columns to standard format
                for std_name, alternatives in col_mapping.items():
                    for alt in alternatives:
                        if alt in df.columns:
                            df.rename(columns={alt: std_name}, inplace=True)
                            break

            # Clean data
            df = df.dropna(subset=required_cols)
            print(f"Loaded {len(df)} records")

            # Encode categorical features
            df['Passenger_Encoded'] = df['Passenger_Type'].map(self.passenger_type_map)
            df['Road_Encoded'] = df['Road_Condition'].map(self.road_condition_map)
            df['Weather_Encoded'] = df['Weather_Condition'].map(self.weather_map)

            # Handle time encoding
            df['Time_Encoded'] = df['Time_of_Day'].apply(
                lambda x: self._encode_time_of_day(x) if ':' in str(x)
                else self.time_of_day_map.get(x, 1)
            )

            # Drop rows with encoding errors
            df = df.dropna(subset=['Passenger_Encoded', 'Road_Encoded', 'Weather_Encoded', 'Time_Encoded'])

            # Create feature matrix with interaction features
            X = np.column_stack([
                df['Distance'].values,
                df['Passenger_Encoded'].values,
                df['Time_Encoded'].values,
                df['Road_Encoded'].values,
                df['Weather_Encoded'].values,
                df['Distance'].values * df['Road_Encoded'].values,  # Interaction
                df['Distance'].values * df['Time_Encoded'].values  # Interaction
            ])

            y = df['Fare_Charged'].values

            # Scale features and train model
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True

            # Calculate dataset-based pricing statistics
            self._calculate_dataset_statistics(df, X, y)

            # Show data statistics
            print(f"\nDataset Statistics:")
            print(f"  Distance range: {df['Distance'].min():.2f} - {df['Distance'].max():.2f} km")
            print(f"  Fare range: ₱{df['Fare_Charged'].min():.2f} - ₱{df['Fare_Charged'].max():.2f}")
            print(f"  Passenger types: {df['Passenger_Type'].value_counts().to_dict()}")

        except FileNotFoundError:
            print(f"Error: File '{csv_path}' not found. Using synthetic data instead.")
            self._generate_training_data()
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            print("Using synthetic data instead.")
            self._generate_training_data()

    def _calculate_dataset_statistics(self, df, X, y):
        """
        Calculate pricing components from the actual dataset using regression analysis
        This makes fare breakdown reflect the real data patterns
        """
        try:
            # Calculate base fare (intercept) and per-km rate using simple linear regression
            from sklearn.linear_model import LinearRegression

            # Simple distance vs fare regression
            lr = LinearRegression()
            lr.fit(df[['Distance']].values, y)
            base_fare = max(0, lr.intercept_)
            per_km_rate = lr.coef_[0]

            self.dataset_stats['base_fare'] = round(base_fare, 2)
            self.dataset_stats['per_km_rate'] = round(per_km_rate, 2)

            # Calculate passenger type effects
            for ptype in ['Regular', 'Student', 'Senior', 'PWD']:
                ptype_data = df[df['Passenger_Type'] == ptype]
                if len(ptype_data) > 0:
                    # Calculate average fare per km for this passenger type
                    ptype_fare_per_km = (ptype_data['Fare_Charged'] / ptype_data['Distance']).mean()
                    regular_data = df[df['Passenger_Type'] == 'Regular']
                    if len(regular_data) > 0:
                        regular_fare_per_km = (regular_data['Fare_Charged'] / regular_data['Distance']).mean()
                        # Calculate percentage difference
                        if ptype != 'Regular':
                            discount = ((regular_fare_per_km - ptype_fare_per_km) / regular_fare_per_km) * 100
                            self.dataset_stats['passenger_discounts'][ptype] = round(max(0, discount), 1)

            # Calculate time of day effects
            regular_midday = df[(df['Passenger_Type'] == 'Regular') & (df['Time_of_Day'] == 'Midday')]
            if len(regular_midday) > 0:
                midday_fare_per_km = (regular_midday['Fare_Charged'] / regular_midday['Distance']).mean()

                for time in ['Morning Peak', 'Afternoon Peak', 'Evening']:
                    time_data = df[(df['Passenger_Type'] == 'Regular') & (df['Time_of_Day'] == time)]
                    if len(time_data) > 0:
                        time_fare_per_km = (time_data['Fare_Charged'] / time_data['Distance']).mean()
                        surcharge = ((time_fare_per_km - midday_fare_per_km) / midday_fare_per_km) * 100
                        self.dataset_stats['time_surcharges'][time] = round(max(0, surcharge), 1)

            # Calculate road condition effects
            paved_data = df[df['Road_Condition'] == 'Paved']
            if len(paved_data) > 0:
                paved_fare_per_km = (paved_data['Fare_Charged'] / paved_data['Distance']).mean()

                for road in ['Unpaved', 'Difficult']:
                    road_data = df[df['Road_Condition'] == road]
                    if len(road_data) > 0:
                        road_fare_per_km = (road_data['Fare_Charged'] / road_data['Distance']).mean()
                        surcharge = ((road_fare_per_km - paved_fare_per_km) / paved_fare_per_km) * 100
                        self.dataset_stats['road_surcharges'][road] = round(max(0, surcharge), 1)

            # Calculate weather effects
            clear_data = df[df['Weather_Condition'] == 'Clear']
            if len(clear_data) > 0:
                clear_fare_per_km = (clear_data['Fare_Charged'] / clear_data['Distance']).mean()

                for weather in ['Overcast', 'Rainy']:
                    weather_data = df[df['Weather_Condition'] == weather]
                    if len(weather_data) > 0:
                        weather_fare_per_km = (weather_data['Fare_Charged'] / weather_data['Distance']).mean()
                        surcharge = ((weather_fare_per_km - clear_fare_per_km) / clear_fare_per_km) * 100
                        self.dataset_stats['weather_surcharges'][weather] = round(max(0, surcharge), 1)

            print(f"\n📊 Calculated Pricing from Dataset:")
            print(f"  Base fare: ₱{self.dataset_stats['base_fare']:.2f}")
            print(f"  Per km rate: ₱{self.dataset_stats['per_km_rate']:.2f}/km")
            print(f"  Passenger discounts: {self.dataset_stats['passenger_discounts']}")
            print(f"  Time surcharges: {self.dataset_stats['time_surcharges']}")
            print(f"  Road surcharges: {self.dataset_stats['road_surcharges']}")
            print(f"  Weather surcharges: {self.dataset_stats['weather_surcharges']}")

        except Exception as e:
            print(f"Warning: Could not calculate dataset statistics: {e}")
            print("Using default pricing structure")

    def _encode_time_of_day(self, time_str):
        """Convert HH:MM time to time of day category"""
        try:
            hour = int(time_str.split(':')[0])
            if 6 <= hour < 9:
                return self.time_of_day_map['Morning Peak']
            elif 9 <= hour < 16:
                return self.time_of_day_map['Midday']
            elif 16 <= hour < 19:
                return self.time_of_day_map['Afternoon Peak']
            else:
                return self.time_of_day_map['Evening']
        except:
            return self.time_of_day_map['Midday']  # Default

    def predict_fare(self, distance, passenger_type='Regular',
                     time_of_day='12:00', road_condition='Paved',
                     weather='Clear'):
        """
        Predict fair fare for a habal-habal ride

        Parameters:
        -----------
        distance : float
            Trip distance in kilometers
        passenger_type : str
            Type of passenger: 'Regular', 'Student', 'Senior', 'PWD'
        time_of_day : str
            Time in HH:MM format
        road_condition : str
            Road quality: 'Paved', 'Unpaved', 'Difficult'
        weather : str
            Weather condition: 'Clear', 'Overcast', 'Rainy'

        Returns:
        --------
        dict : Contains predicted_fare, confidence_interval, and neighbors_info
        """
        if not self.is_trained:
            raise Exception("Model not trained yet")

        # Encode categorical features
        passenger_encoded = self.passenger_type_map.get(passenger_type, 0)
        time_encoded = self._encode_time_of_day(time_of_day)
        road_encoded = self.road_condition_map.get(road_condition, 0)
        weather_encoded = self.weather_map.get(weather, 0)

        # Create feature vector with interaction features
        features = np.array([[
            distance,
            passenger_encoded,
            time_encoded,
            road_encoded,
            weather_encoded,
            distance * road_encoded,  # Interaction
            distance * time_encoded  # Interaction
        ]])

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict fare
        predicted_fare = self.model.predict(features_scaled)[0]

        # Get neighbor information for confidence
        distances, indices = self.model.kneighbors(features_scaled)
        neighbor_distances = distances[0]

        # Calculate confidence interval based on neighbor variance
        avg_distance = np.mean(neighbor_distances)
        confidence_range = max(5, avg_distance * 10)  # Minimum ±5 PHP

        return {
            'predicted_fare': round(predicted_fare, 2),
            'confidence_lower': round(predicted_fare - confidence_range, 2),
            'confidence_upper': round(predicted_fare + confidence_range, 2),
            'confidence_level': 'High' if avg_distance < 0.5 else 'Medium' if avg_distance < 1.0 else 'Low'
        }

    def check_overpricing(self, predicted_fare, actual_fare, threshold=0.20):
        """
        Check if actual fare exceeds predicted fare beyond acceptable threshold

        Parameters:
        -----------
        predicted_fare : float
            AI-predicted fair fare
        actual_fare : float
            Actual fare charged by driver
        threshold : float
            Acceptable deviation (default: 20%)

        Returns:
        --------
        dict : Contains overpricing status and details
        """
        difference = actual_fare - predicted_fare
        percentage_diff = (difference / predicted_fare) * 100 if predicted_fare > 0 else 0

        is_overpriced = percentage_diff > (threshold * 100)

        return {
            'is_overpriced': is_overpriced,
            'difference': round(difference, 2),
            'percentage_difference': round(percentage_diff, 2),
            'threshold_percentage': threshold * 100,
            'status': 'Overpriced' if is_overpriced else 'Fair' if percentage_diff > -10 else 'Good Deal'
        }

    def get_fare_breakdown(self, distance, passenger_type='Regular',
                           time_of_day='12:00', road_condition='Paved',
                           weather='Clear'):
        """Get detailed breakdown of fare components based on dataset statistics"""
        base_fare = self.dataset_stats['base_fare']
        per_km = self.dataset_stats['per_km_rate']
        base_cost = base_fare + (distance * per_km)

        # Get modifiers from dataset statistics
        passenger_discount = self.dataset_stats['passenger_discounts'].get(passenger_type, 0)

        time_hour = int(time_of_day.split(':')[0]) if ':' in time_of_day else 12
        if 5 <= time_hour < 10:
            time_category = 'Morning Peak'
        elif 10 <= time_hour < 14:
            time_category = 'Midday'
        elif 14 <= time_hour < 18:
            time_category = 'Afternoon Peak'
        else:
            time_category = 'Evening'

        time_surcharge = self.dataset_stats['time_surcharges'].get(time_category, 0)
        road_surcharge = self.dataset_stats['road_surcharges'].get(road_condition, 0)
        weather_surcharge = self.dataset_stats['weather_surcharges'].get(weather, 0)

        return {
            'base_fare': base_fare,
            'distance_charge': round(distance * per_km, 2),
            'passenger_discount': passenger_discount,
            'time_surcharge': time_surcharge,
            'road_surcharge': road_surcharge,
            'weather_surcharge': weather_surcharge
        }