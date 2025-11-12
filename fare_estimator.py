import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd


class BusFareEstimator:

    def __init__(self, dataset_path=None, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.model = None  # Will be set after hyperparameter tuning
        self.scaler = StandardScaler()
        self.is_trained = False

        # Store model performance metrics
        self.model_metrics = {
            'train_score': None,
            'test_score': None,
            'cv_mean_score': None,
            'cv_std_score': None,
            'mae': None,
            'rmse': None,
            'r2': None,
            'best_k': None,
            'best_weights': None,
            'best_metric': None
        }

        # Store dataset statistics (calculated from data aggregation, no linear regression)
        self.dataset_stats = {
            'base_fare_city_aircon': 15,
            'base_fare_city_ordinary': 13,
            'base_fare_provincial': 11,
            'per_km_rate_city_aircon': 2.75,
            'per_km_rate_city_ordinary': 2.25,
            'per_km_rate_provincial': 1.9,
            'passenger_discounts': {'Regular': 0, 'Discounted': 20}
        }

        # Feature columns (will be populated after one-hot encoding)
        self.feature_columns = []

        # Store training data for neighbor analysis
        self.X_train_original = None
        self.X_train_scaled = None
        self.y_train = None
        self.train_metadata = None

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
        Load LTFRB bus fare dataset from CSV file with train-test split and hyperparameter tuning
        Uses one-hot encoding for categorical features
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

            # Standardize passenger type naming - consolidate all discounted types
            df['passenger_type'] = df['passenger_type'].replace({
                'Student/Elderly/PWD': 'Discounted',
                'Student': 'Discounted',
                'Elderly': 'Discounted',
                'PWD': 'Discounted'
            })

            # One-hot encode categorical features
            df_encoded = pd.get_dummies(df, columns=['route_type', 'bus_type', 'passenger_type'],
                                        prefix=['route', 'bus', 'passenger'])

            # Create interaction features with one-hot encoded columns
            # Distance x Route interactions
            for col in df_encoded.columns:
                if col.startswith('route_'):
                    df_encoded[f'distance_x_{col}'] = df_encoded['distance_km'] * df_encoded[col]
                elif col.startswith('bus_'):
                    df_encoded[f'distance_x_{col}'] = df_encoded['distance_km'] * df_encoded[col]
                elif col.startswith('passenger_'):
                    df_encoded[f'distance_x_{col}'] = df_encoded['distance_km'] * df_encoded[col]

            # Separate features and target
            feature_cols = [col for col in df_encoded.columns if col != 'fare_php']
            self.feature_columns = feature_cols

            X = df_encoded[feature_cols].values
            y = df_encoded['fare_php'].values

            # Split data into training and testing sets (80:20)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=True
            )

            print(f"\n📊 Dataset Split:")
            print(f"  Training set: {len(X_train)} samples ({(1 - self.test_size) * 100:.0f}%)")
            print(f"  Testing set: {len(X_test)} samples ({self.test_size * 100:.0f}%)")
            print(f"  Feature dimensions: {X_train.shape[1]} features (with one-hot encoding + interactions)")

            # Scale features using training data only
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Store training data for neighbor analysis
            train_indices = X_train.shape[0]
            self.X_train_original = X_train
            self.X_train_scaled = X_train_scaled
            self.y_train = y_train

            # Store metadata for display (reconstruct from original df)
            # We need to map back to original data
            df['index_col'] = range(len(df))
            train_df, test_df = train_test_split(df, test_size=self.test_size,
                                                 random_state=self.random_state, shuffle=True)
            self.train_metadata = train_df[['route_type', 'bus_type', 'passenger_type', 'distance_km']].reset_index(
                drop=True)

            # Hyperparameter tuning with GridSearchCV
            print(f"\n🔍 Performing Hyperparameter Tuning...")
            self._tune_hyperparameters(X_train_scaled, y_train)

            # Train final model with best parameters
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True

            # Cross-validation on training data
            print(f"\n🔄 Performing 5-Fold Cross-Validation...")
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train,
                                        cv=5, scoring='r2')
            self.model_metrics['cv_mean_score'] = round(cv_scores.mean(), 4)
            self.model_metrics['cv_std_score'] = round(cv_scores.std(), 4)

            print(f"  CV R² Scores: {[f'{s:.4f}' for s in cv_scores]}")
            print(
                f"  Mean CV R²: {self.model_metrics['cv_mean_score']:.4f} (±{self.model_metrics['cv_std_score']:.4f})")

            # Evaluate model performance on test set
            self._evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test)

            # Calculate dataset-based pricing statistics (using aggregation, not linear regression)
            self._calculate_dataset_statistics(df)

            # Show data statistics
            print(f"\n📈 Dataset Statistics:")
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

    def _tune_hyperparameters(self, X_train_scaled, y_train):
        """
        Use GridSearchCV to find optimal KNN hyperparameters
        """
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        grid_search = GridSearchCV(
            KNeighborsRegressor(),
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train_scaled, y_train)

        # Store best parameters
        self.model = grid_search.best_estimator_
        self.model_metrics['best_k'] = grid_search.best_params_['n_neighbors']
        self.model_metrics['best_weights'] = grid_search.best_params_['weights']
        self.model_metrics['best_metric'] = grid_search.best_params_['metric']

        print(f"  Best Parameters Found:")
        print(f"    k (neighbors): {self.model_metrics['best_k']}")
        print(f"    weights: {self.model_metrics['best_weights']}")
        print(f"    distance metric: {self.model_metrics['best_metric']}")
        print(f"    Best CV MAE: ₱{-grid_search.best_score_:.2f}")

    def _evaluate_model(self, X_train_scaled, y_train, X_test_scaled, y_test):
        """
        Evaluate model performance on both training and testing sets
        """
        try:
            # Training set performance
            y_train_pred = self.model.predict(X_train_scaled)
            train_score = self.model.score(X_train_scaled, y_train)

            # Testing set performance
            y_test_pred = self.model.predict(X_test_scaled)
            test_score = self.model.score(X_test_scaled, y_test)

            # Calculate error metrics on test set
            mae = mean_absolute_error(y_test, y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            r2 = r2_score(y_test, y_test_pred)

            # Store metrics
            self.model_metrics['train_score'] = round(train_score, 4)
            self.model_metrics['test_score'] = round(test_score, 4)
            self.model_metrics['mae'] = round(mae, 2)
            self.model_metrics['rmse'] = round(rmse, 2)
            self.model_metrics['r2'] = round(r2, 4)

            print(f"\n🎯 Model Performance Metrics:")
            print(f"  Training R² Score: {self.model_metrics['train_score']:.4f}")
            print(f"  Testing R² Score: {self.model_metrics['test_score']:.4f}")
            print(f"  Mean Absolute Error (MAE): ₱{self.model_metrics['mae']:.2f}")
            print(f"  Root Mean Squared Error (RMSE): ₱{self.model_metrics['rmse']:.2f}")
            print(f"  R² Score: {self.model_metrics['r2']:.4f}")

            # Check for overfitting
            if train_score - test_score > 0.1:
                print(
                    f"  ⚠️ Warning: Possible overfitting detected (train-test score gap: {train_score - test_score:.4f})")
            else:
                print(f"  ✅ Model generalization looks good!")

        except Exception as e:
            print(f"Warning: Could not evaluate model: {e}")

    def _calculate_dataset_statistics(self, df):
        """
        Calculate pricing components from actual LTFRB dataset using aggregation (no linear regression)
        """
        try:
            # Calculate base fare for different categories (minimum fare for short trips)
            short_trips = df[df['distance_km'] <= 5]

            for route in ['City']:
                for bus in ['Aircon', 'Ordinary']:
                    subset = short_trips[(short_trips['route_type'] == route) &
                                         (short_trips['bus_type'] == bus) &
                                         (short_trips['passenger_type'] == 'Regular')]

                    if len(subset) > 0:
                        base_key = f'base_fare_{route.lower()}_{bus.lower()}'.replace(' ', '_')
                        self.dataset_stats[base_key] = round(subset['fare_php'].min(), 2)

            # Provincial base fare
            subset = short_trips[(short_trips['route_type'] == 'Provincial') &
                                 (short_trips['bus_type'] == 'Ordinary') &
                                 (short_trips['passenger_type'] == 'Regular')]
            if len(subset) > 0:
                self.dataset_stats['base_fare_provincial'] = round(subset['fare_php'].min(), 2)

            # Calculate per-km rates (average fare per km for longer trips)
            longer_trips = df[df['distance_km'] > 5]

            for route in ['City']:
                for bus in ['Aircon', 'Ordinary']:
                    subset = longer_trips[(longer_trips['route_type'] == route) &
                                          (longer_trips['bus_type'] == bus) &
                                          (longer_trips['passenger_type'] == 'Regular')]

                    if len(subset) > 0:
                        rate_key = f'per_km_rate_{route.lower()}_{bus.lower()}'.replace(' ', '_')
                        base_key = f'base_fare_{route.lower()}_{bus.lower()}'.replace(' ', '_')
                        base_fare = self.dataset_stats.get(base_key, 15)

                        # Calculate incremental rate: (fare - base) / distance
                        avg_rate = ((subset['fare_php'] - base_fare) / subset['distance_km']).mean()
                        self.dataset_stats[rate_key] = round(avg_rate, 2)

            # Provincial per-km rate
            subset = longer_trips[(longer_trips['route_type'] == 'Provincial') &
                                  (longer_trips['bus_type'] == 'Ordinary') &
                                  (longer_trips['passenger_type'] == 'Regular')]
            if len(subset) > 0:
                base_fare = self.dataset_stats['base_fare_provincial']
                avg_rate = ((subset['fare_php'] - base_fare) / subset['distance_km']).mean()
                self.dataset_stats['per_km_rate_provincial'] = round(avg_rate, 2)

            # Calculate discount percentages (direct comparison)
            regular_fares = df[df['passenger_type'] == 'Regular']['fare_php'].mean()
            discounted_fares = df[df['passenger_type'] == 'Discounted']['fare_php'].mean()

            if regular_fares > 0 and discounted_fares > 0:
                discount_pct = ((regular_fares - discounted_fares) / regular_fares) * 100
                self.dataset_stats['passenger_discounts']['Discounted'] = round(discount_pct, 1)

            print(f"\n💰 LTFRB Pricing Structure (from data aggregation):")
            print(f"  City Aircon Base: ₱{self.dataset_stats.get('base_fare_city_aircon', 0):.2f}")
            print(f"  City Ordinary Base: ₱{self.dataset_stats.get('base_fare_city_ordinary', 0):.2f}")
            print(f"  Provincial Base: ₱{self.dataset_stats.get('base_fare_provincial', 0):.2f}")
            print(f"  Discounts: {self.dataset_stats['passenger_discounts']}")

        except Exception as e:
            print(f"Warning: Could not calculate dataset statistics: {e}")

    def _encode_input_features(self, distance, route_type, bus_type, passenger_type):
        """
        Encode input features using one-hot encoding to match training data
        """
        # Create a feature vector matching the training columns
        feature_dict = {}

        # Add distance
        feature_dict['distance_km'] = distance

        # One-hot encode route_type
        for col in self.feature_columns:
            if col.startswith('route_'):
                route_name = col.replace('route_', '')
                feature_dict[col] = 1 if route_name == route_type else 0
            elif col.startswith('bus_'):
                bus_name = col.replace('bus_', '')
                feature_dict[col] = 1 if bus_name == bus_type else 0
            elif col.startswith('passenger_'):
                passenger_name = col.replace('passenger_', '')
                feature_dict[col] = 1 if passenger_name == passenger_type else 0
            elif col.startswith('distance_x_'):
                # Handle interaction features
                base_col = col.replace('distance_x_', '')
                if base_col in feature_dict:
                    feature_dict[col] = distance * feature_dict[base_col]
                else:
                    feature_dict[col] = 0

        # Convert to array in correct order
        features = np.array([[feature_dict.get(col, 0) for col in self.feature_columns]])
        return features

    def predict_fare(self, distance, route_type='City',
                     bus_type='Ordinary', passenger_type='Regular'):
        """
        Predict bus fare using KNN model with improved confidence intervals
        """
        if not self.is_trained:
            raise Exception("Model not trained yet. Please ensure bus_fare_ltfrb_data.csv is loaded.")

        # Encode features
        features = self._encode_input_features(distance, route_type, bus_type, passenger_type)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict fare
        predicted_fare = self.model.predict(features_scaled)[0]

        # Get neighbor information
        distances_to_neighbors, indices = self.model.kneighbors(features_scaled)
        neighbor_fares = self.y_train[indices[0]]
        neighbor_distances = distances_to_neighbors[0]

        # Calculate confidence interval based on neighbor variance
        std_dev = np.std(neighbor_fares)
        confidence_lower = predicted_fare - (1.96 * std_dev)  # 95% CI
        confidence_upper = predicted_fare + (1.96 * std_dev)

        # Determine confidence level based on std dev
        if std_dev < 2:
            confidence_level = 'High'
        elif std_dev < 5:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'

        # Get neighbor details for display
        neighbor_info = []
        for i, idx in enumerate(indices[0]):
            neighbor_info.append({
                'route_type': self.train_metadata.iloc[idx]['route_type'],
                'bus_type': self.train_metadata.iloc[idx]['bus_type'],
                'passenger_type': self.train_metadata.iloc[idx]['passenger_type'],
                'distance_km': round(self.train_metadata.iloc[idx]['distance_km'], 2),
                'fare': round(self.y_train[idx], 2),
                'similarity_distance': round(neighbor_distances[i], 4)
            })

        return {
            'predicted_fare': round(predicted_fare, 2),
            'confidence_lower': round(max(0, confidence_lower), 2),
            'confidence_upper': round(confidence_upper, 2),
            'confidence_level': confidence_level,
            'std_dev': round(std_dev, 2),
            'neighbor_info': neighbor_info
        }

    def check_overpricing(self, predicted_fare, actual_fare, threshold=0.15):
        """
        Check if actual fare exceeds LTFRB-predicted fare
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
        """
        Get detailed breakdown of fare components based on KNN neighbor analysis
        """
        if not self.is_trained:
            raise Exception("Model not trained yet.")

        # Get prediction with neighbor info
        prediction = self.predict_fare(distance, route_type, bus_type, passenger_type)

        # Analyze neighbors to decompose fare
        neighbor_fares = [n['fare'] for n in prediction['neighbor_info']]
        avg_neighbor_fare = np.mean(neighbor_fares)

        # Estimate base fare from dataset statistics
        if route_type == 'City':
            if bus_type == 'Aircon':
                base_fare = self.dataset_stats.get('base_fare_city_aircon', 15)
                per_km = self.dataset_stats.get('per_km_rate_city_aircon', 2.75)
            else:
                base_fare = self.dataset_stats.get('base_fare_city_ordinary', 13)
                per_km = self.dataset_stats.get('per_km_rate_city_ordinary', 2.25)
        else:
            base_fare = self.dataset_stats.get('base_fare_provincial', 11)
            per_km = self.dataset_stats.get('per_km_rate_provincial', 1.9)

        # Calculate components
        distance_charge = distance * per_km
        passenger_discount = self.dataset_stats['passenger_discounts'].get(passenger_type, 0)

        # Bus premium (only for Deluxe buses in Provincial routes)
        bus_premium = 0
        if bus_type == 'Deluxe':
            bus_premium = 10

        # Calculate estimated formula fare
        formula_fare = base_fare + distance_charge + bus_premium
        if passenger_discount > 0:
            formula_fare = formula_fare * (1 - passenger_discount / 100)

        return {
            'predicted_fare': prediction['predicted_fare'],
            'formula_estimate': round(formula_fare, 2),
            'base_fare': base_fare,
            'distance_charge': round(distance_charge, 2),
            'passenger_discount': passenger_discount,
            'bus_premium': bus_premium,
            'route_type': route_type,
            'bus_type': bus_type,
            'neighbor_avg': round(avg_neighbor_fare, 2),
            'knn_adjustment': round(prediction['predicted_fare'] - formula_fare, 2)
        }

    def get_model_performance(self):
        """
        Return the stored model performance metrics
        """
        return self.model_metrics