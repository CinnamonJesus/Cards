"""
Main graphical user interface for the Blackjack Strategy Simulator.
This version automatically optimizes CPU usage and includes critical fixes
for threading stability to prevent crashes between simulations.
"""
from __future__ import annotations
import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QGridLayout,
    QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, QTextEdit, QMessageBox,
    QProgressBar, QButtonGroup, QInputDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont

import counting
import shoe
import simulator
import strategy
import bayesian_predictor
import decision_advisor

class SimulationWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, shoe_dict: dict[str, int], num_threads: int):
        super().__init__()
        self.shoe_dict = shoe_dict
        self.num_threads = num_threads

    def run(self):
        try:
            sim = simulator.FastSimulator(self.shoe_dict)
            results = sim.run(total_rounds=250_000, num_threads=8)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(f"A critical error occurred in the simulation engine:\n{e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blackjack Strategy Simulator (Stable)")
        self.resize(1200, 800)

        self.shoe = shoe.Shoe(decks=8)
        self.counters = {
            "Hi-Lo": counting.HiLoCount(), "Zen": counting.ZenCount(),
            "Wong Halves": counting.WongHalves(), "Omega II": counting.Omega2Count(),
        }
        self.strategy_advisor = strategy.StrategyAdvisor()
        
        self.simulation_thread: QThread | None = None
        self.simulation_worker: SimulationWorker | None = None
        self.dealer_hole_card_placeholder: str | None = None
        
        self._init_round_state()
        self._init_ui()
        self.update_displays()

    def _init_round_state(self):
        self.selected_cards: dict[str, list[str]] = {"player": [], "dealer": [], "burned": []}
        self.action_history: list[tuple[str, str]] = []
        self.round_card_count: int = 0
        self.current_selection_mode: str = "player"
        self.last_sim_results: dict[str, float] | None = None
        self.dealer_hole_card_placeholder = None

    def _init_ui(self):
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
        layout = QVBoxLayout()
        mode_group = QGroupBox("Card Input Mode")
        mode_layout = QHBoxLayout()
        self.radio_player = QRadioButton("Player")
        self.radio_dealer = QRadioButton("Dealer")
        self.radio_burned = QRadioButton("Burned")
        self.radio_player.setChecked(True)
        
        self.mode_button_group = QButtonGroup(self)
        self.mode_button_group.addButton(self.radio_player)
        self.mode_button_group.addButton(self.radio_dealer)
        self.mode_button_group.addButton(self.radio_burned)
        self.mode_button_group.buttonToggled.connect(self._on_radio_toggled)

        mode_layout.addWidget(self.radio_player)
        mode_layout.addWidget(self.radio_dealer)
        mode_layout.addWidget(self.radio_burned)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        card_grid_group = QGroupBox("Select Cards")
        grid_layout = QGridLayout()
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
        self.run_sim_button = QPushButton("Run Sim & End Round")
        self.run_sim_button.clicked.connect(self._run_simulation_and_end_round)
        self.undo_button = QPushButton("Undo Last")
        self.undo_button.clicked.connect(self._undo_last_card)
        self.reset_button = QPushButton("Reset Shoe")
        self.reset_button.clicked.connect(self._reset_shoe)
        
        control_layout.addWidget(self.run_sim_button, 0, 0)
        control_layout.addWidget(self.undo_button, 0, 1)
        control_layout.addWidget(self.reset_button, 1, 0, 1, 2)
        layout.addLayout(control_layout)
        return layout

    def _create_info_panel(self) -> QVBoxLayout:
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
            self.count_labels[f"{name}_tc"] = QLabel("0.0")
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

    def _on_radio_toggled(self, button, checked):
        if not checked: return
        mode_map = {self.radio_player: "player", self.radio_dealer: "dealer", self.radio_burned: "burned"}
        self.current_selection_mode = mode_map.get(button, "player")

    def _card_button_clicked(self, card_code: str):
        mode = self.current_selection_mode
        if self.round_card_count == 0: mode = "player"
        elif self.round_card_count == 1: mode = "dealer"
        elif self.round_card_count == 2: mode = "player"
        
        self._add_card_to_game(card_code, mode)

        if self.round_card_count == 3:
            try:
                placeholder_card = self.shoe.draw_random_card()
                self.dealer_hole_card_placeholder = placeholder_card
                self._add_card_to_game(placeholder_card, "hole")
            except ValueError as e:
                QMessageBox.warning(self, "Shoe Error", str(e))

    def _add_card_to_game(self, card_code: str, mode: str):
        try:
            if card_code != self.dealer_hole_card_placeholder:
                self.shoe.remove_card(card_code)
            
            for counter in self.counters.values(): counter.update(card_code)
            
            if mode in self.selected_cards:
                 self.selected_cards[mode].append(card_code)
            
            self.action_history.append((card_code, mode))
            self.round_card_count += 1
        except (ValueError, KeyError) as e:
            QMessageBox.warning(self, "Card Error", str(e))
            return
        self.update_displays()

    def _undo_last_card(self):
        if not self.action_history: return
        last_card, last_mode = self.action_history.pop()
        
        try:
            self.shoe.restore_card(last_card)
            if last_mode in self.selected_cards and last_card in self.selected_cards[last_mode]:
                 self.selected_cards[last_mode].remove(last_card)
            
            for counter in self.counters.values(): counter.undo(last_card)
            self.round_card_count -= 1

            if last_card == self.dealer_hole_card_placeholder:
                self.dealer_hole_card_placeholder = None
        except (ValueError, KeyError) as e:
            QMessageBox.critical(self, "Undo Error", f"Could not undo action: {e}")
            self.action_history.append((last_card, last_mode))
            return
        self.update_displays()

    def _run_simulation_and_end_round(self):
        if self.simulation_thread and self.simulation_thread.isRunning():
            QMessageBox.warning(self, "Simulation in Progress", "A simulation is already running.")
            return
        self._start_simulation_thread()

    def _start_simulation_thread(self):
        self.run_sim_button.setEnabled(False)
        self.run_sim_button.setText("Simulating...")
        self.sim_output.setText("Running high-performance simulation...")

        num_cores_to_use = max(1, os.cpu_count() - 1)
        
        self.simulation_thread = QThread()
        self.simulation_worker = SimulationWorker(self.shoe.get_remaining_cards(), num_cores_to_use)
        self.simulation_worker.moveToThread(self.simulation_thread)

        self.simulation_thread.started.connect(self.simulation_worker.run)
        self.simulation_worker.finished.connect(self._on_simulation_finished)
        self.simulation_worker.error.connect(self._on_simulation_error)
        
        self.simulation_worker.finished.connect(self.simulation_thread.quit)
        self.simulation_worker.finished.connect(self.simulation_worker.deleteLater)
        self.simulation_thread.finished.connect(self.simulation_thread.deleteLater)
        # BUG FIX: Connect the finished signal to the cleanup slot.
        self.simulation_thread.finished.connect(self._cleanup_thread_references)
        
        self.simulation_thread.start()

    def _on_simulation_finished(self, results: dict[str, float]):
        self.last_sim_results = results
        self.run_sim_button.setEnabled(True)
        self.run_sim_button.setText("Run Sim & End Round")
        
        self._prompt_for_hole_card()
        
        self._init_round_state() 
        self.last_sim_results = results
        self.update_displays()
        self.sim_output.append("\n\n--- End of Round ---\nReady for next hand.")

    def _prompt_for_hole_card(self):
        if not self.dealer_hole_card_placeholder: return
        all_cards = list(self.card_buttons.keys())
        actual_card, ok = QInputDialog.getItem(self, "End of Round", 
                                               "Enter the Dealer's actual hole card:", all_cards, 0, False)
        if ok and actual_card:
            try:
                self.shoe.restore_card(self.dealer_hole_card_placeholder)
                for counter in self.counters.values(): counter.undo(self.dealer_hole_card_placeholder)
                self.shoe.remove_card(actual_card)
                for counter in self.counters.values(): counter.update(actual_card)
            except (ValueError, KeyError) as e:
                QMessageBox.critical(self, "Shoe Correction Error", f"Failed to correct shoe state: {e}")

    def _on_simulation_error(self, error_message: str):
        QMessageBox.critical(self, "Simulation Error", error_message)
        self.sim_output.setText(f"Error during simulation:\n{error_message}")
        self.run_sim_button.setEnabled(True)
        self.run_sim_button.setText("Run Sim & End Round")

    def _cleanup_thread_references(self):
        """
        BUG FIX: This method clears the references to the thread and worker
        objects, preventing the 'wrapped C/C++ object has been deleted' error.
        """
        self.simulation_thread = None
        self.simulation_worker = None

    def _reset_shoe(self):
        reply = QMessageBox.question(self, "Confirm Reset", "Reset the entire shoe?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No: return
        self.shoe.reset_shoe()
        for counter in self.counters.values(): counter.reset()
        self._init_round_state()
        self.sim_output.clear()
        self.update_displays()

    def update_displays(self):
        display_text = ""
        for role, cards in self.selected_cards.items():
            display_text += f"{role.title():<7}: {' '.join(cards)}\n"
        if self.dealer_hole_card_placeholder:
            display_text += f"Hole   : [Placeholder: {self.dealer_hole_card_placeholder}]\n"
        self.selected_cards_display.setText(display_text)
        
        for card_code, button in self.card_buttons.items():
            button.setEnabled(self.shoe.cards.get(card_code, 0) > 0)

        decks_remaining = self.shoe.decks_remaining()
        for name, counter in self.counters.items():
            rc, tc = counter.running_count, counter.true_count(decks_remaining)
            self.count_labels[f"{name}_rc"].setText(f"{rc:g}")
            self.count_labels[f"{name}_tc"].setText(f"{tc:.0f}")

        self.cards_remaining_label.setText(f"{self.shoe.total_cards} / {self.shoe.initial_card_count}")
        self.penetration_bar.setValue(int(self.shoe.get_penetration() * 100))
        
        try:
            player_hand = self.selected_cards["player"]
            dealer_upcard = self.selected_cards["dealer"][0] if self.selected_cards["dealer"] else None
            
            if len(player_hand) >= 2 and dealer_upcard:
                hilo_tc = self.counters["Hi-Lo"].true_count(decks_remaining)
                advice = decision_advisor.recommend_action(player_hand, dealer_upcard, hilo_tc)
                self.advisor_output.setText(advice)
            else:
                self.advisor_output.setText("Enter Player (2) and Dealer (1) cards for advice.")
        except Exception as e:
            self.advisor_output.setText(f"Advisor Error: {e}")

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
            pass
        else:
            self.sim_output.setText("Enter cards and run simulation to get betting advice.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())