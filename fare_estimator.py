import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd


class BusFareEstimator:

    def __init__(self, k=5, dataset_path=None):
        self.k = k
        self.model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        self.scaler = StandardScaler()
        self.is_trained = False

        # Store dataset statistics for fare breakdown
        self.dataset_stats = {
            'base_fare_city_aircon': 15,
            'base_fare_city_ordinary': 13,
            'base_fare_provincial': 11,
            'per_km_rate_city_aircon': 2.75,
            'per_km_rate_city_ordinary': 2.25,
            'per_km_rate_provincial': 1.9,
            'passenger_discounts': {'Regular': 0, 'Student': 20, 'Elderly': 20, 'PWD': 20}
        }

        # Feature encoding mappings
        self.route_type_map = {
            'City': 0,
            'Provincial': 1
        }

        self.bus_type_map = {
            'Ordinary': 0,
            'Aircon': 1,
            'Deluxe': 2,
            'Super Deluxe': 3,
            'Luxury': 4
        }

        self.passenger_type_map = {
            'Regular': 0,
            'Student': 1,
            'Elderly': 2,
            'PWD': 3,
            'Student/Elderly/PWD': 1  # For compatibility with CSV format
        }

        # Load dataset
        if dataset_path:
            self._load_dataset(dataset_path)
        else:
            # Try to load the bus fare dataset
            try:
                self._load_dataset('bus_fare_ltfrb_data.csv')
            except:
                print("Error: bus_fare_ltfrb_data.csv not found!")
                self.is_trained = False

    def _load_dataset(self, csv_path):
        """
        Load LTFRB bus fare dataset from CSV file

        Expected CSV columns:
        - route_type: str (City/Provincial)
        - bus_type: str (Aircon/Ordinary/Deluxe/Super Deluxe/Luxury)
        - distance_km: float (kilometers)
        - passenger_type: str (Regular/Student/Elderly/PWD)
        - fare_php: float (PHP)
        """
        try:
            print(f"Loading LTFRB bus fare dataset from: {csv_path}")
            df = pd.read_csv(csv_path)

            # Validate required columns
            required_cols = ['route_type', 'bus_type', 'distance_km', 'passenger_type', 'fare_php']

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Clean data
            df = df.dropna(subset=required_cols)
            print(f"Loaded {len(df)} fare records from LTFRB dataset")

            # Encode categorical features
            df['Route_Encoded'] = df['route_type'].map(self.route_type_map)
            df['Bus_Encoded'] = df['bus_type'].map(self.bus_type_map)
            df['Passenger_Encoded'] = df['passenger_type'].map(self.passenger_type_map)

            # Drop rows with encoding errors
            df = df.dropna(subset=['Route_Encoded', 'Bus_Encoded', 'Passenger_Encoded'])

            # Create feature matrix with interaction features
            X = np.column_stack([
                df['distance_km'].values,
                df['Route_Encoded'].values,
                df['Bus_Encoded'].values,
                df['Passenger_Encoded'].values,
                df['distance_km'].values * df['Bus_Encoded'].values,  # Distance-bus interaction
                df['distance_km'].values * df['Route_Encoded'].values  # Distance-route interaction
            ])

            y = df['fare_php'].values

            # Scale features and train model
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True

            # Calculate dataset-based pricing statistics
            self._calculate_dataset_statistics(df)

            # Show data statistics
            print(f"\nDataset Statistics:")
            print(f"  Distance range: {df['distance_km'].min():.2f} - {df['distance_km'].max():.2f} km")
            print(f"  Fare range: ₱{df['fare_php'].min():.2f} - ₱{df['fare_php'].max():.2f}")
            print(f"  Route types: {df['route_type'].value_counts().to_dict()}")
            print(f"  Bus types: {df['bus_type'].value_counts().to_dict()}")
            print(f"  Passenger types: {df['passenger_type'].value_counts().to_dict()}")

        except FileNotFoundError:
            print(f"Error: File '{csv_path}' not found.")
            raise
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def _calculate_dataset_statistics(self, df):
        """
        Calculate pricing components from the actual LTFRB dataset
        """
        try:
            from sklearn.linear_model import LinearRegression

            # Calculate base fare and per-km rates for different bus types
            for route in ['City']:
                for bus in ['Aircon', 'Ordinary']:
                    subset = df[(df['route_type'] == route) &
                                (df['bus_type'] == bus) &
                                (df['passenger_type'] == 'Regular')]

                    if len(subset) > 0:
                        lr = LinearRegression()
                        lr.fit(subset[['distance_km']].values, subset['fare_php'].values)

                        base_key = f'base_fare_{route.lower()}_{bus.lower()}'.replace(' ', '_')
                        rate_key = f'per_km_rate_{route.lower()}_{bus.lower()}'.replace(' ', '_')

                        self.dataset_stats[base_key] = round(max(0, lr.intercept_), 2)
                        self.dataset_stats[rate_key] = round(lr.coef_[0], 2)

            # Calculate provincial rates
            subset = df[(df['route_type'] == 'Provincial') &
                        (df['bus_type'] == 'Ordinary') &
                        (df['passenger_type'] == 'Regular')]
            if len(subset) > 0:
                lr = LinearRegression()
                lr.fit(subset[['distance_km']].values, subset['fare_php'].values)
                self.dataset_stats['base_fare_provincial'] = round(max(0, lr.intercept_), 2)
                self.dataset_stats['per_km_rate_provincial'] = round(lr.coef_[0], 2)

            # Calculate discount percentages
            regular_fares = df[df['passenger_type'] == 'Regular']['fare_php'].mean()
            discounted_fares = df[df['passenger_type'] == 'Student/Elderly/PWD']['fare_php'].mean()

            if regular_fares > 0:
                discount_pct = ((regular_fares - discounted_fares) / regular_fares) * 100
                self.dataset_stats['passenger_discounts']['Student'] = round(discount_pct, 1)
                self.dataset_stats['passenger_discounts']['Elderly'] = round(discount_pct, 1)
                self.dataset_stats['passenger_discounts']['PWD'] = round(discount_pct, 1)

            print(f"\n📊 LTFRB Pricing Structure:")
            print(f"  City Aircon Base: ₱{self.dataset_stats.get('base_fare_city_aircon', 0):.2f}")
            print(f"  City Ordinary Base: ₱{self.dataset_stats.get('base_fare_city_ordinary', 0):.2f}")
            print(f"  Provincial Base: ₱{self.dataset_stats.get('base_fare_provincial', 0):.2f}")
            print(f"  Discounts: {self.dataset_stats['passenger_discounts']}")

        except Exception as e:
            print(f"Warning: Could not calculate dataset statistics: {e}")

    def predict_fare(self, distance, route_type='City',
                     bus_type='Ordinary', passenger_type='Regular'):
        """
        Predict bus fare based on LTFRB standards

        Parameters:
        -----------
        distance : float
            Trip distance in kilometers
        route_type : str
            Route type: 'City' or 'Provincial'
        bus_type : str
            Bus type: 'Ordinary', 'Aircon', 'Deluxe', 'Super Deluxe', 'Luxury'
        passenger_type : str
            Passenger type: 'Regular', 'Student', 'Elderly', 'PWD'

        Returns:
        --------
        dict : Contains predicted_fare, confidence_interval, and neighbors_info
        """
        if not self.is_trained:
            raise Exception("Model not trained yet. Please ensure bus_fare_ltfrb_data.csv is loaded.")

        # Encode categorical features
        route_encoded = self.route_type_map.get(route_type, 0)
        bus_encoded = self.bus_type_map.get(bus_type, 0)
        passenger_encoded = self.passenger_type_map.get(passenger_type, 0)

        # Create feature vector with interaction features
        features = np.array([[
            distance,
            route_encoded,
            bus_encoded,
            passenger_encoded,
            distance * bus_encoded,  # Distance-bus interaction
            distance * route_encoded  # Distance-route interaction
        ]])

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict fare
        predicted_fare = self.model.predict(features_scaled)[0]

        # Get neighbor information for confidence
        distances_to_neighbors, indices = self.model.kneighbors(features_scaled)
        neighbor_distances = distances_to_neighbors[0]

        # Calculate confidence interval based on neighbor variance
        avg_distance = np.mean(neighbor_distances)
        confidence_range = max(3, avg_distance * 8)  # Minimum ±3 PHP

        return {
            'predicted_fare': round(predicted_fare, 2),
            'confidence_lower': round(predicted_fare - confidence_range, 2),
            'confidence_upper': round(predicted_fare + confidence_range, 2),
            'confidence_level': 'High' if avg_distance < 0.3 else 'Medium' if avg_distance < 0.6 else 'Low'
        }

    def check_overpricing(self, predicted_fare, actual_fare, threshold=0.15):
        """
        Check if actual fare exceeds LTFRB-predicted fare

        Parameters:
        -----------
        predicted_fare : float
            LTFRB-based predicted fare
        actual_fare : float
            Actual fare charged
        threshold : float
            Acceptable deviation (default: 15%)

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
            'status': 'Overpriced' if is_overpriced else 'Fair' if percentage_diff > -5 else 'Good Deal'
        }

    def get_fare_breakdown(self, distance, route_type='City',
                           bus_type='Ordinary', passenger_type='Regular'):
        """Get detailed breakdown of fare components based on LTFRB standards"""

        # Determine base fare and per km rate
        if route_type == 'City':
            if bus_type == 'Aircon':
                base_fare = self.dataset_stats.get('base_fare_city_aircon', 15)
                per_km = self.dataset_stats.get('per_km_rate_city_aircon', 2.75)
            else:  # Ordinary
                base_fare = self.dataset_stats.get('base_fare_city_ordinary', 13)
                per_km = self.dataset_stats.get('per_km_rate_city_ordinary', 2.25)
        else:  # Provincial
            base_fare = self.dataset_stats.get('base_fare_provincial', 11)
            per_km = self.dataset_stats.get('per_km_rate_provincial', 1.9)

        base_cost = base_fare
        distance_charge = distance * per_km

        # Get passenger discount
        passenger_discount = self.dataset_stats['passenger_discounts'].get(passenger_type, 0)

        # Bus type premium (for deluxe/luxury buses)
        bus_premium = 0
        if bus_type in ['Deluxe', 'Super Deluxe', 'Luxury']:
            bus_premium = 10 if bus_type == 'Deluxe' else 15 if bus_type == 'Super Deluxe' else 25

        return {
            'base_fare': base_fare,
            'distance_charge': round(distance_charge, 2),
            'passenger_discount': passenger_discount,
            'bus_premium': bus_premium,
            'route_type': route_type,
            'bus_type': bus_type
        }