import tkinter as tk
from tkinter import ttk
from fare_estimator import BusFareEstimator


class FareRequestScreen:
    def __init__(self, parent_frame, app_controller):
        self.parent_frame = parent_frame
        self.app_controller = app_controller

        # Initialize AI Fare Estimator with LTFRB data
        try:
            self.fare_estimator = BusFareEstimator(k=5, dataset_path='bus_fare_ltfrb_data.csv')
        except Exception as e:
            print(f"Error loading fare estimator: {e}")
            self.fare_estimator = None

        # Variables
        self.distance = tk.StringVar(value="")
        self.route_type = tk.StringVar(value="City")
        self.bus_type = tk.StringVar(value="Ordinary")
        self.passenger_type = tk.StringVar(value="Regular")
        self.actual_fare = tk.StringVar(value="")

        self.predicted_fare = "₱--.--"

        # Main container with scrollbar
        self.canvas = tk.Canvas(parent_frame, bg="#1a1a1a", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(parent_frame, orient="vertical", command=self.canvas.yview)

        self.main_frame = tk.Frame(self.canvas, bg="#1a1a1a")

        self.main_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

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
            text="LTFRB Fare Calculator",
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
            text="🤖 Powered by K-Nearest Neighbors (KNN) Algorithm\nBased on Official LTFRB Fare Matrix Standards",
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
        self._create_label(input_frame, "📏 Distance (km)")
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

        # Route Type selector
        self._create_label(input_frame, "🛣️ Route Type")
        route_frame = tk.Frame(input_frame, bg="#1a1a1a")
        route_frame.pack(fill=tk.X, pady=(0, 15))

        route_options = ['City', 'Provincial']
        for option in route_options:
            btn = tk.Button(
                route_frame,
                text=option,
                command=lambda o=option: self.select_route_type(o),
                font=("Arial", 10),
                bg="#2a2a2a",
                fg="white",
                activebackground="#3a3a3a",
                relief=tk.FLAT,
                cursor="hand2"
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            if option == 'City':
                btn.config(bg="#2196f3", activebackground="#42a5f5")
                self.selected_route_btn = btn

        # Bus Type selector
        self._create_label(input_frame, "🚌 Bus Type")
        bus_frame = tk.Frame(input_frame, bg="#1a1a1a")
        bus_frame.pack(fill=tk.X, pady=(0, 15))

        bus_options = ['Ordinary', 'Aircon', 'Deluxe', 'Super Deluxe', 'Luxury']
        self.bus_buttons = []
        for i, option in enumerate(bus_options):
            if i % 3 == 0:
                row_frame = tk.Frame(bus_frame, bg="#1a1a1a")
                row_frame.pack(fill=tk.X, pady=2)

            btn = tk.Button(
                row_frame,
                text=option,
                command=lambda o=option: self.select_bus_type(o),
                font=("Arial", 9),
                bg="#2a2a2a",
                fg="white",
                activebackground="#3a3a3a",
                relief=tk.FLAT,
                cursor="hand2",
                width=10
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            self.bus_buttons.append(btn)

            if option == 'Ordinary':
                btn.config(bg="#2196f3", activebackground="#42a5f5")
                self.selected_bus_btn = btn

        # Passenger Type selector
        self._create_label(input_frame, "👤 Passenger Type")
        passenger_frame = tk.Frame(input_frame, bg="#1a1a1a")
        passenger_frame.pack(fill=tk.X, pady=(0, 15))

        passenger_options = ['Regular', 'Student', 'Elderly', 'PWD']
        for option in passenger_options:
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
                btn.config(bg="#2196f3", activebackground="#42a5f5")
                self.selected_passenger_btn = btn

        # Calculate button
        calculate_btn = tk.Button(
            input_frame,
            text="🔮 Calculate Fare",
            font=("Arial", 12, "bold"),
            bg="#4caf50",
            fg="white",
            activebackground="#5cbf60",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.calculate_fare,
            pady=12
        )
        calculate_btn.pack(fill=tk.X, pady=(10, 15))

        # AI Prediction Results
        self.results_frame = tk.Frame(self.main_frame, bg="#2a2a2a")
        self.results_frame.pack(fill=tk.X, padx=20, pady=15)

        results_title = tk.Label(
            self.results_frame,
            text="LTFRB-Based Fare Calculation",
            font=("Arial", 12, "bold"),
            bg="#2a2a2a",
            fg="#2196f3"
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
            text="Enter trip details to calculate",
            font=("Arial", 9),
            bg="#2a2a2a",
            fg="#999999"
        )
        self.confidence_label.pack(pady=(0, 10))

        # Fare breakdown
        self.breakdown_frame = tk.Frame(self.results_frame, bg="#2a2a2a")
        self.breakdown_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Compliance Check Section
        check_frame = tk.Frame(self.main_frame, bg="#1a1a1a")
        check_frame.pack(fill=tk.X, padx=20, pady=(0, 15))

        check_title = tk.Label(
            check_frame,
            text="⚖️ Check Fare Compliance",
            font=("Arial", 12, "bold"),
            bg="#1a1a1a",
            fg="#2196f3"
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
            text="🔍 Check Compliance",
            font=("Arial", 11, "bold"),
            bg="#ff9800",
            fg="white",
            activebackground="#ffa726",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.check_compliance,
            pady=10
        )
        check_btn.pack(fill=tk.X)

        # Compliance result
        self.compliance_result = tk.Frame(check_frame, bg="#1a1a1a")
        self.compliance_result.pack(fill=tk.X, pady=(10, 0))

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

    def select_route_type(self, route):
        """Update route type selection"""
        self.route_type.set(route)

        # Reset button styles
        for widget in self.selected_route_btn.master.winfo_children():
            widget.config(bg="#2a2a2a", activebackground="#3a3a3a")

        # Highlight selected
        for widget in self.selected_route_btn.master.winfo_children():
            if widget['text'] == route:
                widget.config(bg="#2196f3", activebackground="#42a5f5")
                self.selected_route_btn = widget

    def select_bus_type(self, bus):
        """Update bus type selection"""
        self.bus_type.set(bus)

        # Reset all bus button styles
        for btn in self.bus_buttons:
            btn.config(bg="#2a2a2a", activebackground="#3a3a3a")

        # Highlight selected
        for btn in self.bus_buttons:
            if btn['text'] == bus:
                btn.config(bg="#2196f3", activebackground="#42a5f5")
                self.selected_bus_btn = btn

    def select_passenger_type(self, ptype):
        """Update passenger type selection"""
        self.passenger_type.set(ptype)

        # Reset button styles
        for widget in self.selected_passenger_btn.master.winfo_children():
            widget.config(bg="#2a2a2a", activebackground="#3a3a3a")

        # Highlight selected
        for widget in self.selected_passenger_btn.master.winfo_children():
            if widget['text'] == ptype:
                widget.config(bg="#2196f3", activebackground="#42a5f5")
                self.selected_passenger_btn = widget

    def calculate_fare(self):
        """Calculate fare using LTFRB-based AI model"""
        if not self.fare_estimator:
            self.show_error("Fare estimator not initialized. Please check bus_fare_ltfrb_data.csv")
            return

        try:
            distance = float(self.distance.get()) if self.distance.get() else 0

            if distance <= 0:
                self.show_error("Please enter a valid distance")
                return

            # Get prediction from AI model
            prediction = self.fare_estimator.predict_fare(
                distance=distance,
                route_type=self.route_type.get(),
                bus_type=self.bus_type.get(),
                passenger_type=self.passenger_type.get()
            )

            # Get fare breakdown
            breakdown = self.fare_estimator.get_fare_breakdown(
                distance=distance,
                route_type=self.route_type.get(),
                bus_type=self.bus_type.get(),
                passenger_type=self.passenger_type.get()
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
                ("Route Type", breakdown['route_type']),
                ("Bus Type", breakdown['bus_type']),
                ("Base Fare", f"₱{breakdown['base_fare']:.2f}"),
                ("Distance Charge", f"₱{breakdown['distance_charge']:.2f}"),
            ]

            if breakdown['passenger_discount'] > 0:
                breakdown_items.append(("Passenger Discount", f"-{breakdown['passenger_discount']}%"))

            if breakdown['bus_premium'] > 0:
                breakdown_items.append(("Bus Premium", f"+{breakdown['bus_premium']}%"))

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
                    fg="#2196f3",
                    anchor="e"
                ).pack(side=tk.RIGHT)

        except ValueError:
            self.show_error("Invalid input. Please check your entries.")
        except Exception as e:
            self.show_error(f"Error: {str(e)}")

    def check_compliance(self):
        """Check if actual fare complies with LTFRB standards"""
        if not self.fare_estimator:
            self.show_error("Fare estimator not initialized")
            return

        try:
            if not self.predicted_fare_label.cget("text").startswith("₱"):
                self.show_error("Please calculate fare first")
                return

            predicted = float(self.predicted_fare_label.cget("text").replace("₱", ""))
            actual = float(self.actual_fare.get())

            result = self.fare_estimator.check_overpricing(predicted, actual)

            # Clear previous result
            for widget in self.compliance_result.winfo_children():
                widget.destroy()

            # Show result
            result_frame = tk.Frame(self.compliance_result, bg="#2a2a2a", padx=15, pady=15)
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
                    text=f"⚠️ Exceeds LTFRB standard by more than {result['threshold_percentage']:.0f}%\nYou may report to LTFRB Hotline: 1342",
                    font=("Arial", 9),
                    bg="#2a2a2a",
                    fg="#ff5252",
                    justify="center"
                )
                warning.pack(pady=5)
            else:
                ok_label = tk.Label(
                    result_frame,
                    text="✓ Fare is within LTFRB acceptable range",
                    font=("Arial", 9),
                    bg="#2a2a2a",
                    fg="#4caf50"
                )
                ok_label.pack(pady=5)

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
        self.canvas.destroy()
        self.scrollbar.destroy()