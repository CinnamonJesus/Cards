"""
Main graphical user interface for the Blackjack Strategy Simulator.

This stable version provides a comprehensive and crash-free interface for:
- Manually inputting cards with a guided, sequential workflow.
- Displaying real-time card counting metrics (Hi-Lo, Zen, Wong Halves, Omega II).
- Running high-speed, stable Monte Carlo simulations to calculate Expected Value (EV).
- Providing strategic advice based on basic strategy and simulation results.
- Visualizing game state information like deck penetration and card counts.
"""
from __future__ import annotations
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QGridLayout,
    QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, QTextEdit, QMessageBox,
    QProgressBar, QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor

# Import the refined backend modules
import counting
import shoe
import simulator
import strategy
import bayesian_predictor
import decision_advisor

# --- Worker for Offloading Simulations ---

class SimulationWorker(QObject):
    """
    A worker object that runs the simulation in a separate thread to avoid
    freezing the GUI. Emits results or an error message upon completion.
    """
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, shoe_dict: dict[str, int]):
        super().__init__()
        self.shoe_dict = shoe_dict

    def run(self):
        """Executes the simulation and emits the results or any exceptions."""
        try:
            # Ensure the simulator is instantiated and run within the try block
            sim = simulator.FastSimulator(self.shoe_dict)
            results = sim.run(total_rounds=500_000, num_threads=4)
            self.finished.emit(results)
        except Exception as e:
            # Catch any exception from the simulation and emit an error signal
            self.error.emit(f"A critical error occurred in the simulation engine:\n{e}")

# --- Main Application Window ---

class MainWindow(QMainWindow):
    """
    The main window for the Blackjack Simulator application. This version includes
    robust error handling, stable threading, and corrected state management.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blackjack Strategy Simulator (Stable Version)")
        self.resize(1200, 800)

        # --- Initialize Backend Components ---
        self.shoe = shoe.Shoe(decks=8)
        self.counters = {
            "Hi-Lo": counting.HiLoCount(),
            "Zen": counting.ZenCount(),
            "Wong Halves": counting.WongHalves(),
            "Omega II": counting.Omega2Count(),
        }
        self.strategy_advisor = strategy.StrategyAdvisor()
        
        # --- State Management ---
        self.simulation_thread: QThread | None = None
        self.simulation_worker: SimulationWorker | None = None
        self._init_round_state()

        # --- Build UI ---
        self._init_ui()
        self.update_displays()

    def _init_round_state(self):
        """Initializes or resets the state for a new round of betting."""
        # SIMPLIFIED STATE: Using simple lists, not lists of lists.
        self.selected_cards: dict[str, list[str]] = {"player": [], "dealer": [], "burned": []}
        self.action_history: list[tuple[str, str]] = [] # Stores (card_code, mode)
        self.round_card_count: int = 0
        self.current_selection_mode: str = "player"
        self.last_sim_results: dict[str, float] | None = None

    def _init_ui(self):
        """Initializes the main user interface layout and widgets."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = self._create_card_input_panel()
        middle_panel = self._create_info_panel()
        right_panel = self._create_strategy_panel()

        main_layout.addLayout(left_panel, 0)
        main_layout.addLayout(middle_panel, 1)
        main_layout.addLayout(right_panel, 2)
    
    def _create_card_input_panel(self) -> QVBoxLayout:
        """Creates the left panel for card selection and input."""
        layout = QVBoxLayout()
        
        mode_group = QGroupBox("Card Input Mode")
        mode_layout = QHBoxLayout()
        self.radio_player = QRadioButton("Player")
        self.radio_dealer = QRadioButton("Dealer")
        self.radio_burned = QRadioButton("Burned")
        self.radio_player.setChecked(True) # Default to player
        
        self.mode_button_group = QButtonGroup(self)
        self.mode_button_group.addButton(self.radio_player)
        self.mode_button_group.addButton(self.radio_dealer)
        self.mode_button_group.addButton(self.radio_burned)
        
        self.radio_player.toggled.connect(lambda: self._on_radio_toggled("player"))
        self.radio_dealer.toggled.connect(lambda: self._on_radio_toggled("dealer"))
        self.radio_burned.toggled.connect(lambda: self._on_radio_toggled("burned"))

        mode_layout.addWidget(self.radio_player)
        mode_layout.addWidget(self.radio_dealer)
        mode_layout.addWidget(self.radio_burned)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        card_grid_group = QGroupBox("Select Cards")
        grid_layout = QGridLayout()
        grid_layout.setSpacing(4)
        self.card_buttons: dict[str, QPushButton] = {}
        ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        suits = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
        suit_colors = {"S": "black", "H": "red", "D": "red", "C": "black"}

        for i, rank in enumerate(ranks):
            for j, (suit_code, suit_char) in enumerate(suits.items()):
                card_code = f"{rank}{suit_code}"
                button = QPushButton(f"{rank}{suit_char}")
                button.setFixedSize(40, 30)
                button.setStyleSheet(f"color: {suit_colors[suit_code]}; font-weight: bold;")
                button.clicked.connect(lambda _, c=card_code: self._card_button_clicked(c))
                grid_layout.addWidget(button, i, j)
                self.card_buttons[card_code] = button
        card_grid_group.setLayout(grid_layout)
        layout.addWidget(card_grid_group)

        control_layout = QGridLayout()
        self.run_sim_button = QPushButton("Run Simulation")
        self.run_sim_button.clicked.connect(self._run_simulation)
        self.undo_button = QPushButton("Undo Last")
        self.undo_button.clicked.connect(self._undo_last_card)
        self.reset_button = QPushButton("Reset Shoe")
        self.reset_button.clicked.connect(self._reset_shoe)
        
        control_layout.addWidget(self.run_sim_button, 0, 0)
        control_layout.addWidget(self.undo_button, 0, 1)
        control_layout.addWidget(self.reset_button, 1, 0, 1, 2) # Span reset button
        layout.addLayout(control_layout)

        return layout

    def _create_info_panel(self) -> QVBoxLayout:
        """Creates the middle panel for displaying game state and counts."""
        layout = QVBoxLayout()
        self.selected_cards_display = QTextEdit()
        self.selected_cards_display.setReadOnly(True)
        self.selected_cards_display.setFont(QFont("Courier New", 10))
        self.selected_cards_display.setMinimumHeight(120)
        layout.addWidget(QLabel("Current Hand:"))
        layout.addWidget(self.selected_cards_display)

        counts_group = QGroupBox("Card Counting")
        counts_layout = QGridLayout()
        self.count_labels: dict[str, QLabel] = {}
        for i, name in enumerate(self.counters.keys()):
            counts_layout.addWidget(QLabel(f"{name} RC:"), i, 0)
            self.count_labels[f"{name}_rc"] = QLabel("0")
            counts_layout.addWidget(self.count_labels[f"{name}_rc"], i, 1)
            counts_layout.addWidget(QLabel(f"{name} TC:"), i, 2)
            self.count_labels[f"{name}_tc"] = QLabel("0.00")
            counts_layout.addWidget(self.count_labels[f"{name}_tc"], i, 3)
        counts_group.setLayout(counts_layout)
        layout.addWidget(counts_group)

        shoe_group = QGroupBox("Shoe Status")
        shoe_layout = QGridLayout()
        shoe_layout.addWidget(QLabel("Cards Remaining:"), 0, 0)
        self.cards_remaining_label = QLabel("")
        shoe_layout.addWidget(self.cards_remaining_label, 0, 1)
        shoe_layout.addWidget(QLabel("Penetration:"), 1, 0)
        self.penetration_bar = QProgressBar()
        shoe_layout.addWidget(self.penetration_bar, 1, 1)
        shoe_group.setLayout(shoe_layout)
        layout.addWidget(shoe_group)
        layout.addStretch()
        return layout

    def _create_strategy_panel(self) -> QVBoxLayout:
        """Creates the right panel for displaying strategic advice."""
        layout = QVBoxLayout()
        advisor_group = QGroupBox("Basic Strategy Decision Advisor")
        advisor_layout = QVBoxLayout()
        self.advisor_output = QLabel("Enter Player and Dealer cards for advice.")
        self.advisor_output.setWordWrap(True)
        self.advisor_output.setStyleSheet("font-size: 16px; font-weight: bold; color: #336699;")
        advisor_layout.addWidget(self.advisor_output)
        advisor_group.setLayout(advisor_layout)
        layout.addWidget(advisor_group)

        sim_strat_group = QGroupBox("Simulation & Betting Strategy")
        sim_strat_layout = QVBoxLayout()
        self.sim_output = QTextEdit()
        self.sim_output.setReadOnly(True)
        self.sim_output.setFont(QFont("Courier New", 10))
        sim_strat_layout.addWidget(self.sim_output)
        sim_strat_group.setLayout(sim_strat_layout)
        layout.addWidget(sim_strat_group)
        return layout

    # --- Event Handlers and Logic ---

    def _on_radio_toggled(self, mode: str):
        """Updates the current selection mode when a radio button is clicked."""
        if self.radio_player.isChecked():
            self.current_selection_mode = "player"
        elif self.radio_dealer.isChecked():
            self.current_selection_mode = "dealer"
        elif self.radio_burned.isChecked():
            self.current_selection_mode = "burned"

    def _card_button_clicked(self, card_code: str):
        """Handles clicks on any of the 52 card buttons."""
        try:
            self.shoe.remove_card(card_code)
            
            # Add card to the correct hand based on mode
            mode = self.current_selection_mode
            self.selected_cards[mode].append(card_code)
            
            # Update counters and history
            for counter in self.counters.values():
                counter.update(card_code)
            self.action_history.append((card_code, mode))
            self.round_card_count += 1

        except (ValueError, KeyError) as e:
            QMessageBox.warning(self, "Card Error", str(e))
            return

        self.update_displays()

    def _undo_last_card(self):
        """Reverts the last card action."""
        if not self.action_history:
            QMessageBox.information(self, "Undo", "No actions to undo.")
            return
        
        last_card, last_mode = self.action_history.pop()
        
        try:
            self.shoe.restore_card(last_card)
            # Use a robust method to remove the last occurrence of the card
            if last_card in self.selected_cards[last_mode]:
                 self.selected_cards[last_mode].remove(last_card)
            
            for counter in self.counters.values():
                counter.undo(last_card)

            self.round_card_count = max(0, self.round_card_count - 1)
        except (ValueError, KeyError) as e:
            QMessageBox.critical(self, "Undo Error", f"Could not undo action: {e}")
            self.action_history.append((last_card, last_mode)) # Restore history
            return
            
        self.update_displays()

    def _run_simulation(self):
        """Initiates the simulation in a background thread."""
        if self.simulation_thread and self.simulation_thread.isRunning():
            QMessageBox.warning(self, "Simulation in Progress", "A simulation is already running.")
            return

        self.run_sim_button.setEnabled(False)
        self.run_sim_button.setText("Simulating...")
        self.sim_output.setText("Running high-performance simulation, please wait...")

        self.simulation_thread = QThread()
        self.simulation_worker = SimulationWorker(self.shoe.get_remaining_cards())
        self.simulation_worker.moveToThread(self.simulation_thread)

        self.simulation_thread.started.connect(self.simulation_worker.run)
        self.simulation_worker.finished.connect(self._on_simulation_finished)
        self.simulation_worker.error.connect(self._on_simulation_error)
        
        # Clean up thread and worker after they are finished
        self.simulation_worker.finished.connect(self.simulation_thread.quit)
        self.simulation_worker.finished.connect(self.simulation_worker.deleteLater)
        self.simulation_thread.finished.connect(self.simulation_thread.deleteLater)
        
        self.simulation_thread.start()

    def _on_simulation_finished(self, results: dict[str, float]):
        """Handles the results when the simulation worker is done."""
        self.last_sim_results = results
        self.run_sim_button.setEnabled(True)
        self.run_sim_button.setText("Run Simulation")
        
        # After simulation, clear the hand for the next round, but keep shoe state
        self._init_round_state() 
        self.last_sim_results = results # Preserve results for display
        self.update_displays()
        self.sim_output.append("\n\n--- End of Round ---\nReady for next hand input.")


    def _on_simulation_error(self, error_message: str):
        """Handles errors from the simulation worker, preventing crashes."""
        QMessageBox.critical(self, "Simulation Error", error_message)
        self.sim_output.setText(f"Error during simulation:\n{error_message}")
        self.run_sim_button.setEnabled(True)
        self.run_sim_button.setText("Run Simulation")

    def _reset_shoe(self):
        """Resets the entire application to its initial state."""
        reply = QMessageBox.question(self, "Confirm Reset", 
                                     "Are you sure you want to reset the entire shoe?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return

        self.shoe.reset_shoe()
        for counter in self.counters.values():
            counter.reset()
        
        self._init_round_state()
        self.sim_output.clear()
        self.update_displays()

    def update_displays(self):
        """Updates all text labels, progress bars, and strategy outputs."""
        # --- Update Card Displays ---
        display_text = ""
        for role, cards in self.selected_cards.items():
            display_text += f"{role.title():<7}: {' '.join(cards)}\n"
        self.selected_cards_display.setText(display_text)
        
        # --- Update Card Buttons ---
        for card_code, button in self.card_buttons.items():
            button.setEnabled(self.shoe.cards.get(card_code, 0) > 0)

        # --- Update Counters and Shoe Status ---
        decks_remaining = self.shoe.decks_remaining()
        for name, counter in self.counters.items():
            rc, tc = counter.running_count, counter.true_count(decks_remaining)
            self.count_labels[f"{name}_rc"].setText(f"{rc:g}")
            self.count_labels[f"{name}_tc"].setText(f"{tc:.2f}")

        self.cards_remaining_label.setText(f"{self.shoe.total_cards} / {self.shoe.initial_card_count}")
        self.penetration_bar.setValue(int(self.shoe.get_penetration() * 100))
        
        # --- Update Decision Advisor (with safety checks) ---
        try:
            player_hand = self.selected_cards["player"]
            dealer_upcard = self.selected_cards["dealer"][0] if self.selected_cards["dealer"] else None
            
            if len(player_hand) >= 2 and dealer_upcard:
                omega_tc = self.counters["Omega II"].true_count(decks_remaining)
                advice = decision_advisor.recommend_action(player_hand, dealer_upcard, omega_tc)
                self.advisor_output.setText(advice)
            else:
                self.advisor_output.setText("Enter Player (2) and Dealer (1) cards for advice.")
        except Exception as e:
            self.advisor_output.setText(f"Advisor Error: {e}")

        # --- Update Simulation/Strategy Output ---
        if self.last_sim_results:
            try:
                dealer_upcard = self.selected_cards["dealer"][0] if self.selected_cards["dealer"] else None
                bayes_totals = None
                if dealer_upcard:
                    bayes_totals = bayesian_predictor.dealer_total_probabilities(
                        dealer_upcard, self.shoe.get_remaining_cards(), simulations=5000)
                
                recommendations = self.strategy_advisor.generate_recommendations(
                    self.shoe, self.counters, self.last_sim_results, bayes_totals)
                self.sim_output.setText("\n".join(recommendations))
            except Exception as e:
                self.sim_output.setText(f"Strategy Display Error: {e}")
        elif not self.run_sim_button.isEnabled():
            # Keep "Simulating..." message
            pass
        else:
            self.sim_output.setText("Enter cards and run simulation to get betting advice.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
