import tkinter as tk
from request import FareRequestScreen


class MenuScreen:
    def __init__(self, parent_frame, app_controller):
        self.parent_frame = parent_frame
        self.app_controller = app_controller

        # Main container
        self.main_frame = tk.Frame(parent_frame, bg="#1a1a1a")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        header_frame = tk.Frame(self.main_frame, bg="#1a1a1a")
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = tk.Label(
            header_frame,
            text="🚌 Bus Fare Calculator",
            font=("Arial", 32, "bold"),
            bg="#1a1a1a",
            fg="#2196f3"
        )
        title_label.pack()

        subtitle_label = tk.Label(
            header_frame,
            text="LTFRB-Based AI Fare Estimator",
            font=("Arial", 13, "bold"),
            bg="#1a1a1a",
            fg="#4caf50"
        )
        subtitle_label.pack()

        location_label = tk.Label(
            header_frame,
            text="Philippines - City & Provincial Routes",
            font=("Arial", 10),
            bg="#1a1a1a",
            fg="#999999"
        )
        location_label.pack()

        # Info banner
        info_banner = tk.Frame(self.main_frame, bg="#1e3a5f")
        info_banner.pack(fill=tk.X, pady=(10, 20))

        info_text = tk.Label(
            info_banner,
            text="🤖 Powered by K-Nearest Neighbors (KNN)\nBased on LTFRB Official Fare Matrix",
            font=("Arial", 9),
            bg="#1e3a5f",
            fg="white",
            justify="center",
            pady=12
        )
        info_text.pack()

        # Stats section
        stats_frame = tk.Frame(self.main_frame, bg="#1a1a1a")
        stats_frame.pack(fill=tk.X, pady=(0, 20))

        stat_items = [
            ("📊", "1000+", "LTFRB\nRecords"),
            ("🎯", "99%", "Accuracy\nRate"),
            ("⚡", "K=5", "Nearest\nNeighbors")
        ]

        for emoji, value, label in stat_items:
            stat_box = tk.Frame(stats_frame, bg="#2a2a2a")
            stat_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)

            tk.Label(
                stat_box,
                text=emoji,
                font=("Arial", 20),
                bg="#2a2a2a",
                fg="#2196f3",
                pady=5
            ).pack()

            tk.Label(
                stat_box,
                text=value,
                font=("Arial", 14, "bold"),
                bg="#2a2a2a",
                fg="white"
            ).pack()

            tk.Label(
                stat_box,
                text=label,
                font=("Arial", 8),
                bg="#2a2a2a",
                fg="#999999",
                justify="center"
            ).pack(pady=(0, 10))

        # Menu buttons container
        menu_frame = tk.Frame(self.main_frame, bg="#1a1a1a")
        menu_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Button style configuration
        button_config = {
            "font": ("Arial", 13, "bold"),
            "bg": "#2196f3",
            "fg": "white",
            "activebackground": "#42a5f5",
            "activeforeground": "white",
            "relief": tk.FLAT,
            "cursor": "hand2",
            "height": 2
        }

        # Primary button - Calculate Fare
        btn_estimate = tk.Button(
            menu_frame,
            text="🔮  Calculate Bus Fare",
            command=self.calculate_fare,
            **button_config
        )
        btn_estimate.pack(fill=tk.X, pady=8)

        # Secondary buttons with different style
        secondary_config = button_config.copy()
        secondary_config["bg"] = "#2a2a2a"
        secondary_config["activebackground"] = "#3a3a3a"
        secondary_config["font"] = ("Arial", 12)

        btn_check = tk.Button(
            menu_frame,
            text="⚖️  Check Fare Compliance",
            command=self.check_compliance,
            **secondary_config
        )
        btn_check.pack(fill=tk.X, pady=5)

        btn_about = tk.Button(
            menu_frame,
            text="ℹ️  About the System",
            command=self.about_system,
            **secondary_config
        )
        btn_about.pack(fill=tk.X, pady=5)

        btn_report = tk.Button(
            menu_frame,
            text="📝  LTFRB Standards",
            command=self.show_standards,
            **secondary_config
        )
        btn_report.pack(fill=tk.X, pady=5)

        # Footer - Research Info
        footer_frame = tk.Frame(self.main_frame, bg="#1a1a1a")
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(15, 0))

        research_label = tk.Label(
            footer_frame,
            text="🎓 Research Project | Mapúa Malayan Colleges Mindanao",
            font=("Arial", 8),
            bg="#2a2a2a",
            fg="#999999",
            padx=10,
            pady=8
        )
        research_label.pack(fill=tk.X)

        authors_label = tk.Label(
            footer_frame,
            text="Cena • Fuentes • Sevilla",
            font=("Arial", 7),
            bg="#2a2a2a",
            fg="#666666",
            pady=3
        )
        authors_label.pack(fill=tk.X)

    def calculate_fare(self):
        """Switch to fare calculation screen"""
        self.app_controller.show_fare_request()

    def check_compliance(self):
        """Switch to fare calculation screen (same as calculate)"""
        self.app_controller.show_fare_request()

    def about_system(self):
        """Show about dialog"""
        about_window = tk.Toplevel(self.main_frame)
        about_window.title("About the System")
        about_window.geometry("400x500")
        about_window.configure(bg="#1a1a1a")

        # Make it modal
        about_window.transient(self.main_frame)
        about_window.grab_set()

        content_frame = tk.Frame(about_window, bg="#1a1a1a")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(
            content_frame,
            text="🚌 Bus Fare Calculator",
            font=("Arial", 16, "bold"),
            bg="#1a1a1a",
            fg="#2196f3"
        ).pack(pady=(0, 10))

        about_text = """AI-Powered Fare Calculator for Philippine Bus Transportation

This system uses Machine Learning to verify compliance with LTFRB fare regulations.

🤖 Technology:
• K-Nearest Neighbors (KNN) Algorithm
• Supervised Learning Model
• 1000+ LTFRB Training Records
• Real-time Fare Prediction

📊 Features:
• Distance-based fare calculation
• Route type (City/Provincial)
• Bus type classification (Ordinary, Aircon, Deluxe, etc.)
• Passenger discounts (20% for Student/Senior/PWD)
• Fare compliance checking (15% threshold)
• LTFRB standard verification

💥 Research Team:
• Sharmayne Andrea Cena
• Xavier Ignazio Maria Fuentes
• Christina Heliane Sevilla

🏛️ Institution:
Mapúa Malayan Colleges Mindanao
College of Engineering and Architecture

📍 Coverage:
Philippine bus operations nationwide
City and provincial routes
All LTFRB-regulated bus types"""

        text_widget = tk.Text(
            content_frame,
            font=("Arial", 9),
            bg="#2a2a2a",
            fg="#cccccc",
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=15,
            pady=15,
            height=20
        )
        text_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        text_widget.insert("1.0", about_text)
        text_widget.config(state=tk.DISABLED)

        close_btn = tk.Button(
            content_frame,
            text="Close",
            font=("Arial", 11),
            bg="#2196f3",
            fg="white",
            activebackground="#42a5f5",
            relief=tk.FLAT,
            cursor="hand2",
            command=about_window.destroy,
            pady=8
        )
        close_btn.pack(fill=tk.X)

    def show_standards(self):
        """Show LTFRB standards dialog"""
        standards_window = tk.Toplevel(self.main_frame)
        standards_window.title("LTFRB Fare Standards")
        standards_window.geometry("400x450")
        standards_window.configure(bg="#1a1a1a")

        standards_window.transient(self.main_frame)
        standards_window.grab_set()

        content_frame = tk.Frame(standards_window, bg="#1a1a1a")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(
            content_frame,
            text="📝 LTFRB Fare Matrix",
            font=("Arial", 14, "bold"),
            bg="#1a1a1a",
            fg="#2196f3"
        ).pack(pady=(0, 15))

        standards_text = """Official LTFRB Bus Fare Standards

🚌 City Routes (Aircon):
• Base Fare: ₱15.00 (first 5km)
• Additional: ₱2.75 per km

🚌 City Routes (Ordinary):
• Base Fare: ₱13.00 (first 5km)
• Additional: ₱2.25 per km

🚌 Provincial Routes:
• Ordinary: ₱11.00 base + ₱1.90/km
• Deluxe: 25% premium over ordinary
• Super Deluxe: 40% premium
• Luxury: 100% premium

👥 Passenger Discounts:
• Students: 20% discount
• Senior Citizens: 20% discount
• PWD: 20% discount

⚖️ Compliance Standards:
• Maximum allowed variance: ±15%
• Required valid ID for discounts
• Fare matrix posted inside bus

📞 Report Overcharging:
• LTFRB Hotline: 1342
• Email: contact@ltfrb.gov.ph

All fares are regulated by the Land Transportation Franchising and Regulatory Board (LTFRB)."""

        text_widget = tk.Text(
            content_frame,
            font=("Arial", 9),
            bg="#2a2a2a",
            fg="#cccccc",
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=15,
            pady=15,
            height=18
        )
        text_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        text_widget.insert("1.0", standards_text)
        text_widget.config(state=tk.DISABLED)

        close_btn = tk.Button(
            content_frame,
            text="Close",
            font=("Arial", 11),
            bg="#2196f3",
            fg="white",
            activebackground="#42a5f5",
            relief=tk.FLAT,
            cursor="hand2",
            command=standards_window.destroy,
            pady=8
        )
        close_btn.pack(fill=tk.X, side=tk.BOTTOM)

    def destroy(self):
        """Clean up the screen"""
        self.main_frame.destroy()


# ==================== APP CONTROLLER ====================
class BusFareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LTFRB Bus Fare Calculator - Philippines")
        self.root.geometry("450x750")
        self.root.configure(bg="#1a1a1a")

        # Container for switching screens
        self.container = tk.Frame(root, bg="#1a1a1a")
        self.container.pack(fill=tk.BOTH, expand=True)

        self.current_screen = None

        # Show menu screen by default
        self.show_menu()

    def show_menu(self):
        """Display the menu screen"""
        if self.current_screen:
            self.current_screen.destroy()
        self.current_screen = MenuScreen(self.container, self)

    def show_fare_request(self):
        """Display the fare request screen"""
        if self.current_screen:
            self.current_screen.destroy()
        self.current_screen = FareRequestScreen(self.container, self)


if __name__ == "__main__":
    root = tk.Tk()
    app = BusFareApp(root)
    root.mainloop()