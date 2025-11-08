import tkinter as tk
from tkinter import ttk
from fare_estimator import HabalHabalFareEstimator


class RideRequestScreen:
    def __init__(self, parent_frame, app_controller):
        self.parent_frame = parent_frame
        self.app_controller = app_controller

        # Initialize AI Fare Estimator
        # Automatically loads 'habal_habal_dataset.csv' if available
        # Otherwise falls back to synthetic data
        self.fare_estimator = HabalHabalFareEstimator(k=5)

        # Variables
        self.distance = tk.StringVar(value="")
        self.hour = tk.StringVar(value="12")
        self.minute = tk.StringVar(value="00")
        self.am_pm = tk.StringVar(value="PM")
        self.passenger_type = tk.StringVar(value="Regular")
        self.road_condition = tk.StringVar(value="Paved")
        self.weather = tk.StringVar(value="Clear")
        self.actual_fare = tk.StringVar(value="")

        self.eta = "-- min"
        self.predicted_fare = "₱--.--"
        self.fare_status = ""

        # Main container with scrollbar
        canvas = tk.Canvas(parent_frame, bg="#1a1a1a", highlightthickness=0)
        scrollbar = tk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)

        self.main_frame = tk.Frame(canvas, bg="#1a1a1a")

        self.main_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Header with back button
        header_frame = tk.Frame(self.main_frame, bg="#2a2a2a")
        header_frame.pack(fill=tk.X)

        back_btn = tk.Button(
            header_frame,
            text="← Back",
            font=("Arial", 11),
            bg="#2a2a2a",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.go_back,
            padx=15,
            pady=10
        )
        back_btn.pack(side=tk.LEFT)

        header_title = tk.Label(
            header_frame,
            text="AI Fare Estimator",
            font=("Arial", 14, "bold"),
            bg="#2a2a2a",
            fg="white",
            pady=10
        )
        header_title.pack(side=tk.LEFT, padx=20)

        # Info banner
        info_frame = tk.Frame(self.main_frame, bg="#1e3a5f")
        info_frame.pack(fill=tk.X, padx=20, pady=15)

        info_label = tk.Label(
            info_frame,
            text="🤖 Powered by K-Nearest Neighbors (KNN) Algorithm\nPromoting Fair & Transparent Habal-Habal Fares in Davao City",
            font=("Arial", 9),
            bg="#1e3a5f",
            fg="white",
            justify="center",
            pady=10
        )
        info_label.pack()

        # Input section
        input_frame = tk.Frame(self.main_frame, bg="#1a1a1a")
        input_frame.pack(fill=tk.X, padx=20, pady=5)

        # Distance input
        self._create_label(input_frame, "📍 Distance (km)")
        distance_entry = tk.Entry(
            input_frame,
            textvariable=self.distance,
            font=("Arial", 12),
            bg="#2a2a2a",
            fg="white",
            relief=tk.FLAT,
            insertbackground="white"
        )
        distance_entry.pack(fill=tk.X, ipady=8, pady=(0, 15))
        distance_entry.bind('<KeyRelease>', self.update_prediction)

        # Time of day input (12-hour format with AM/PM)
        self._create_label(input_frame, "🕐 Time of Ride")

        time_container = tk.Frame(input_frame, bg="#1a1a1a")
        time_container.pack(fill=tk.X, pady=(0, 15))

        # Hour dropdown
        hour_frame = tk.Frame(time_container, bg="#2a2a2a")
        hour_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        tk.Label(
            hour_frame,
            text="Hour",
            font=("Arial", 8),
            bg="#2a2a2a",
            fg="#999999"
        ).pack()

        hour_dropdown = ttk.Combobox(
            hour_frame,
            textvariable=self.hour,
            values=[f"{i:02d}" for i in range(1, 13)],
            state="readonly",
            font=("Arial", 12),
            width=5
        )
        hour_dropdown.pack(ipady=5)
        hour_dropdown.bind('<<ComboboxSelected>>', self.update_prediction)

        # Minute dropdown
        minute_frame = tk.Frame(time_container, bg="#2a2a2a")
        minute_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        tk.Label(
            minute_frame,
            text="Minute",
            font=("Arial", 8),
            bg="#2a2a2a",
            fg="#999999"
        ).pack()

        minute_dropdown = ttk.Combobox(
            minute_frame,
            textvariable=self.minute,
            values=["00", "15", "30", "45"],
            state="readonly",
            font=("Arial", 12),
            width=5
        )
        minute_dropdown.pack(ipady=5)
        minute_dropdown.bind('<<ComboboxSelected>>', self.update_prediction)

        # AM/PM buttons
        ampm_frame = tk.Frame(time_container, bg="#1a1a1a")
        ampm_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.am_btn = tk.Button(
            ampm_frame,
            text="AM",
            command=lambda: self.select_am_pm("AM"),
            font=("Arial", 11, "bold"),
            bg="#2a2a2a",
            fg="white",
            activebackground="#3a3a3a",
            relief=tk.FLAT,
            cursor="hand2",
            width=5
        )
        self.am_btn.pack(fill=tk.X, pady=(0, 3))

        self.pm_btn = tk.Button(
            ampm_frame,
            text="PM",
            command=lambda: self.select_am_pm("PM"),
            font=("Arial", 11, "bold"),
            bg="#ff6b35",
            fg="white",
            activebackground="#ff8555",
            relief=tk.FLAT,
            cursor="hand2",
            width=5
        )
        self.pm_btn.pack(fill=tk.X)

        # Passenger Type selector
        self._create_label(input_frame, "👤 Passenger Type")
        passenger_frame = tk.Frame(input_frame, bg="#1a1a1a")
        passenger_frame.pack(fill=tk.X, pady=(0, 15))

        passenger_options = ['Regular', 'Student', 'Senior', 'PWD']
        for i, option in enumerate(passenger_options):
            btn = tk.Button(
                passenger_frame,
                text=option,
                command=lambda o=option: self.select_passenger_type(o),
                font=("Arial", 10),
                bg="#2a2a2a",
                fg="white",
                activebackground="#3a3a3a",
                relief=tk.FLAT,
                cursor="hand2",
                width=8
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            if option == 'Regular':
                btn.config(bg="#ff6b35", activebackground="#ff8555")
                self.selected_passenger_btn = btn

        # Road Condition selector
        self._create_label(input_frame, "🛣️ Road Condition")
        road_frame = tk.Frame(input_frame, bg="#1a1a1a")
        road_frame.pack(fill=tk.X, pady=(0, 15))

        road_options = ['Paved', 'Unpaved', 'Difficult']
        for i, option in enumerate(road_options):
            btn = tk.Button(
                road_frame,
                text=option,
                command=lambda o=option: self.select_road_condition(o),
                font=("Arial", 10),
                bg="#2a2a2a",
                fg="white",
                activebackground="#3a3a3a",
                relief=tk.FLAT,
                cursor="hand2"
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            if option == 'Paved':
                btn.config(bg="#ff6b35", activebackground="#ff8555")
                self.selected_road_btn = btn

        # Weather selector
        self._create_label(input_frame, "🌤️ Weather Condition")
        weather_frame = tk.Frame(input_frame, bg="#1a1a1a")
        weather_frame.pack(fill=tk.X, pady=(0, 15))

        weather_options = [('Clear', '☀️'), ('Overcast', '☁️'), ('Rainy', '🌧️')]
        for option, emoji in weather_options:
            btn = tk.Button(
                weather_frame,
                text=f"{emoji} {option}",
                command=lambda o=option: self.select_weather(o),
                font=("Arial", 10),
                bg="#2a2a2a",
                fg="white",
                activebackground="#3a3a3a",
                relief=tk.FLAT,
                cursor="hand2"
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            if option == 'Clear':
                btn.config(bg="#ff6b35", activebackground="#ff8555")
                self.selected_weather_btn = btn

        # Predict button
        predict_btn = tk.Button(
            input_frame,
            text="🔮 Predict Fair Fare",
            font=("Arial", 12, "bold"),
            bg="#4caf50",
            fg="white",
            activebackground="#5cbf60",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.predict_fare,
            pady=12
        )
        predict_btn.pack(fill=tk.X, pady=(10, 15))

        # AI Prediction Results
        self.results_frame = tk.Frame(self.main_frame, bg="#2a2a2a")
        self.results_frame.pack(fill=tk.X, padx=20, pady=15)

        results_title = tk.Label(
            self.results_frame,
            text="AI Prediction Results",
            font=("Arial", 12, "bold"),
            bg="#2a2a2a",
            fg="#ff6b35"
        )
        results_title.pack(pady=(10, 5))

        # Predicted fare display
        self.predicted_fare_label = tk.Label(
            self.results_frame,
            text="₱--.--",
            font=("Arial", 28, "bold"),
            bg="#2a2a2a",
            fg="#4caf50"
        )
        self.predicted_fare_label.pack(pady=5)

        self.confidence_label = tk.Label(
            self.results_frame,
            text="Enter ride details to predict",
            font=("Arial", 9),
            bg="#2a2a2a",
            fg="#999999"
        )
        self.confidence_label.pack(pady=(0, 10))

        # Fare breakdown
        self.breakdown_frame = tk.Frame(self.results_frame, bg="#2a2a2a")
        self.breakdown_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Overpricing Check Section
        check_frame = tk.Frame(self.main_frame, bg="#1a1a1a")
        check_frame.pack(fill=tk.X, padx=20, pady=(0, 15))

        check_title = tk.Label(
            check_frame,
            text="⚖️ Check for Overpricing",
            font=("Arial", 12, "bold"),
            bg="#1a1a1a",
            fg="#ff6b35"
        )
        check_title.pack(pady=(0, 10))

        actual_label = tk.Label(
            check_frame,
            text="Actual Fare Charged (₱)",
            font=("Arial", 10),
            bg="#1a1a1a",
            fg="#cccccc"
        )
        actual_label.pack()

        actual_entry = tk.Entry(
            check_frame,
            textvariable=self.actual_fare,
            font=("Arial", 12),
            bg="#2a2a2a",
            fg="white",
            relief=tk.FLAT,
            insertbackground="white",
            justify="center"
        )
        actual_entry.pack(fill=tk.X, ipady=8, pady=(5, 10))

        check_btn = tk.Button(
            check_frame,
            text="🔍 Check Fare",
            font=("Arial", 11, "bold"),
            bg="#2196f3",
            fg="white",
            activebackground="#42a5f5",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.check_overpricing,
            pady=10
        )
        check_btn.pack(fill=tk.X)

        # Overpricing result
        self.overpricing_result = tk.Frame(check_frame, bg="#1a1a1a")
        self.overpricing_result.pack(fill=tk.X, pady=(10, 0))

    @property
    def time_of_day(self):
        """Convert 12-hour format to 24-hour HH:MM string"""
        hour = int(self.hour.get())
        minute = self.minute.get()
        period = self.am_pm.get()

        # Convert to 24-hour format
        if period == "AM":
            if hour == 12:
                hour = 0
        else:  # PM
            if hour != 12:
                hour += 12

        return f"{hour:02d}:{minute}"

    def _create_label(self, parent, text):
        """Helper to create consistent labels"""
        label = tk.Label(
            parent,
            text=text,
            font=("Arial", 10, "bold"),
            bg="#1a1a1a",
            fg="#cccccc",
            anchor="w"
        )
        label.pack(fill=tk.X, pady=(0, 5))

    def select_am_pm(self, period):
        """Update AM/PM selection"""
        self.am_pm.set(period)

        # Update button styles
        if period == "AM":
            self.am_btn.config(bg="#ff6b35", activebackground="#ff8555")
            self.pm_btn.config(bg="#2a2a2a", activebackground="#3a3a3a")
        else:
            self.am_btn.config(bg="#2a2a2a", activebackground="#3a3a3a")
            self.pm_btn.config(bg="#ff6b35", activebackground="#ff8555")

        self.update_prediction()

    def select_passenger_type(self, ptype):
        """Update passenger type selection"""
        self.passenger_type.set(ptype)

        # Reset button styles
        for widget in self.selected_passenger_btn.master.winfo_children():
            widget.config(bg="#2a2a2a", activebackground="#3a3a3a")

        # Highlight selected
        for widget in self.selected_passenger_btn.master.winfo_children():
            if widget['text'] == ptype:
                widget.config(bg="#ff6b35", activebackground="#ff8555")
                self.selected_passenger_btn = widget

        self.update_prediction()

    def select_road_condition(self, condition):
        """Update road condition selection"""
        self.road_condition.set(condition)

        # Reset button styles
        for widget in self.selected_road_btn.master.winfo_children():
            widget.config(bg="#2a2a2a", activebackground="#3a3a3a")

        # Highlight selected
        for widget in self.selected_road_btn.master.winfo_children():
            if widget['text'] == condition:
                widget.config(bg="#ff6b35", activebackground="#ff8555")
                self.selected_road_btn = widget

        self.update_prediction()

    def select_weather(self, weather):
        """Update weather selection"""
        self.weather.set(weather)

        # Reset button styles
        for widget in self.selected_weather_btn.master.winfo_children():
            widget.config(bg="#2a2a2a", activebackground="#3a3a3a")

        # Highlight selected
        for widget in self.selected_weather_btn.master.winfo_children():
            if weather in widget['text']:
                widget.config(bg="#ff6b35", activebackground="#ff8555")
                self.selected_weather_btn = widget

        self.update_prediction()

    def update_prediction(self, event=None):
        """Auto-update prediction as user types"""
        # Just clear the display, user must click predict button
        pass

    def predict_fare(self):
        """Predict fare using AI model"""
        try:
            distance = float(self.distance.get()) if self.distance.get() else 0

            if distance <= 0:
                self.show_error("Please enter a valid distance")
                return

            # Get prediction from AI model
            prediction = self.fare_estimator.predict_fare(
                distance=distance,
                passenger_type=self.passenger_type.get(),
                time_of_day=self.time_of_day,
                road_condition=self.road_condition.get(),
                weather=self.weather.get()
            )

            # Get fare breakdown
            breakdown = self.fare_estimator.get_fare_breakdown(
                distance=distance,
                passenger_type=self.passenger_type.get(),
                time_of_day=self.time_of_day,
                road_condition=self.road_condition.get(),
                weather=self.weather.get()
            )

            # Update display
            self.predicted_fare_label.config(
                text=f"₱{prediction['predicted_fare']:.2f}"
            )

            confidence_text = f"Confidence: {prediction['confidence_level']} | Range: ₱{prediction['confidence_lower']:.2f} - ₱{prediction['confidence_upper']:.2f}"
            self.confidence_label.config(text=confidence_text)

            # Clear and rebuild breakdown
            for widget in self.breakdown_frame.winfo_children():
                widget.destroy()

            breakdown_items = [
                ("Base Fare", f"₱{breakdown['base_fare']:.2f}"),
                ("Distance Charge", f"₱{breakdown['distance_charge']:.2f}"),
            ]

            if breakdown['passenger_discount'] > 0:
                breakdown_items.append(("Passenger Discount", f"-{breakdown['passenger_discount']}%"))

            breakdown_items.append(("Time Surcharge", f"+{breakdown['time_surcharge']}%"))
            breakdown_items.append(("Road Surcharge", f"+{breakdown['road_surcharge']}%"))
            breakdown_items.append(("Weather Surcharge", f"+{breakdown['weather_surcharge']}%"))

            for label, value in breakdown_items:
                item_frame = tk.Frame(self.breakdown_frame, bg="#2a2a2a")
                item_frame.pack(fill=tk.X, pady=2)

                tk.Label(
                    item_frame,
                    text=label,
                    font=("Arial", 9),
                    bg="#2a2a2a",
                    fg="#cccccc",
                    anchor="w"
                ).pack(side=tk.LEFT)

                tk.Label(
                    item_frame,
                    text=value,
                    font=("Arial", 9, "bold"),
                    bg="#2a2a2a",
                    fg="#ff6b35",
                    anchor="e"
                ).pack(side=tk.RIGHT)

        except ValueError:
            self.show_error("Invalid input. Please check your entries.")
        except Exception as e:
            self.show_error(f"Error: {str(e)}")

    def check_overpricing(self):
        """Check if actual fare is overpriced"""
        try:
            if not self.predicted_fare_label.cget("text").startswith("₱"):
                self.show_error("Please predict fare first")
                return

            predicted = float(self.predicted_fare_label.cget("text").replace("₱", ""))
            actual = float(self.actual_fare.get())

            result = self.fare_estimator.check_overpricing(predicted, actual)

            # Clear previous result
            for widget in self.overpricing_result.winfo_children():
                widget.destroy()

            # Show result
            result_frame = tk.Frame(self.overpricing_result, bg="#2a2a2a", padx=15, pady=15)
            result_frame.pack(fill=tk.X, pady=(10, 0))

            status_colors = {
                'Overpriced': '#ff5252',
                'Fair': '#ff9800',
                'Good Deal': '#4caf50'
            }

            status_label = tk.Label(
                result_frame,
                text=f"Status: {result['status']}",
                font=("Arial", 14, "bold"),
                bg="#2a2a2a",
                fg=status_colors.get(result['status'], 'white')
            )
            status_label.pack()

            diff_text = f"Difference: ₱{result['difference']:.2f} ({result['percentage_difference']:.1f}%)"
            diff_label = tk.Label(
                result_frame,
                text=diff_text,
                font=("Arial", 10),
                bg="#2a2a2a",
                fg="#cccccc"
            )
            diff_label.pack(pady=5)

            if result['is_overpriced']:
                warning = tk.Label(
                    result_frame,
                    text=f"⚠️ Exceeds fair fare by more than {result['threshold_percentage']:.0f}%",
                    font=("Arial", 9),
                    bg="#2a2a2a",
                    fg="#ff5252"
                )
                warning.pack(pady=5)

        except ValueError:
            self.show_error("Please enter valid fare amounts")
        except Exception as e:
            self.show_error(f"Error: {str(e)}")

    def show_error(self, message):
        """Display error message"""
        error_label = tk.Label(
            self.main_frame,
            text=f"⚠️ {message}",
            font=("Arial", 10),
            bg="#ff5252",
            fg="white",
            pady=10
        )
        error_label.place(relx=0.5, rely=0.5, anchor="center")
        self.parent_frame.after(3000, error_label.destroy)

    def go_back(self):
        """Return to menu screen"""
        self.app_controller.show_menu()

    def destroy(self):
        """Clean up the screen"""
        self.main_frame.master.master.destroy()