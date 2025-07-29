"""
Main graphical user interface for the Blackjack Strategy Simulator.
"""
from __future__ import annotations
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QGridLayout,
    QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, QTextEdit, QMessageBox,
    QProgressBar, QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont

import counting, shoe, simulator, strategy
import bayesian_predictor, decision_advisor

class SimulationWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    def __init__(self, shoe_dict: dict[str, int]):
        super().__init__()
        self.shoe_dict = shoe_dict
    def run(self):
        try:
            sim = simulator.FastSimulator(self.shoe_dict)
            results = sim.run()
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(f"Simulation failed: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blackjack Strategy Simulator")
        self.resize(1200, 800)
        self.shoe = shoe.Shoe(decks=8)
        self.counters = {
            "Hi-Lo": counting.HiLoCount(), "Zen": counting.ZenCount(),
            "Wong Halves": counting.WongHalves(), "Omega II": counting.Omega2Count(),
        }
        self.strategy_advisor = strategy.StrategyAdvisor()
        self._init_state()
        self._init_ui()
        self.update_displays()

    def _init_state(self):
        # CORRECTED: All modes now have a list containing one hand list.
        self.selected_cards = {"player": [[]], "dealer": [[]], "burned": [[]]}
        self.player_turn_over = False
        self.action_history = []
        self.round_card_count = 0
        self.current_selection_mode = "player"
        self.last_sim_results = None

    def _init_ui(self):
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.addLayout(self._create_card_input_panel(), 0)
        main_layout.addLayout(self._create_info_panel(), 1)
        main_layout.addLayout(self._create_strategy_panel(), 2)

    def _create_card_input_panel(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        mode_group = QGroupBox("Card Input Mode"); mode_layout = QHBoxLayout()
        self.radio_player, self.radio_dealer, self.radio_burned = QRadioButton("Player"), QRadioButton("Dealer"), QRadioButton("Burned")
        self.mode_button_group = QButtonGroup()
        for btn in [self.radio_player, self.radio_dealer, self.radio_burned]: self.mode_button_group.addButton(btn)
        self.radio_player.toggled.connect(lambda: self._on_radio_toggled("player"))
        self.radio_dealer.toggled.connect(lambda: self._on_radio_toggled("dealer"))
        self.radio_burned.toggled.connect(lambda: self._on_radio_toggled("burned"))
        mode_layout.addWidget(self.radio_player); mode_layout.addWidget(self.radio_dealer); mode_layout.addWidget(self.radio_burned)
        mode_group.setLayout(mode_layout); layout.addWidget(mode_group)

        card_grid_group = QGroupBox("Select Cards"); grid_layout = QGridLayout(); grid_layout.setSpacing(4)
        self.card_buttons = {}
        ranks, suits, colors = "A23456789TJQK", {"S":"♠","H":"♥","D":"♦","C":"♣"}, {"S":"black","H":"red","D":"red","C":"black"}
        for i, r in enumerate(ranks):
            rank_str = "10" if r == "T" else r
            for j, (sc, sch) in enumerate(suits.items()):
                code = f"{rank_str}{sc}"
                btn = QPushButton(f"{rank_str if r != 'T' else '10'}{sch}")
                btn.setFixedSize(40, 30); btn.setStyleSheet(f"color: {colors[sc]}; font-weight: bold;")
                btn.clicked.connect(lambda _, c=code: self._card_button_clicked(c))
                grid_layout.addWidget(btn, i, j); self.card_buttons[code] = btn
        card_grid_group.setLayout(grid_layout); layout.addWidget(card_grid_group)

        action_group = QGroupBox("Player Actions"); action_layout = QGridLayout()
        self.split_button = QPushButton("Split (Disabled)"); self.split_button.setEnabled(False)
        self.stand_button = QPushButton("Stand"); self.stand_button.clicked.connect(self._stand_action)
        action_layout.addWidget(self.split_button, 0, 0); action_layout.addWidget(self.stand_button, 0, 1)
        action_group.setLayout(action_layout); layout.addWidget(action_group)

        control_layout = QHBoxLayout()
        self.run_sim_button = QPushButton("Run Simulation"); self.run_sim_button.clicked.connect(self._run_simulation)
        self.undo_button = QPushButton("Undo Last"); self.undo_button.clicked.connect(self._undo_last_action)
        self.reset_button = QPushButton("Reset Shoe"); self.reset_button.clicked.connect(self._reset_shoe)
        control_layout.addWidget(self.run_sim_button); control_layout.addWidget(self.undo_button)
        layout.addLayout(control_layout); layout.addWidget(self.reset_button)
        return layout

    def _create_info_panel(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        self.selected_cards_display = QTextEdit(); self.selected_cards_display.setReadOnly(True)
        self.selected_cards_display.setFont(QFont("Courier New", 10)); self.selected_cards_display.setMinimumHeight(150)
        layout.addWidget(QLabel("Current Hand(s):")); layout.addWidget(self.selected_cards_display)
        
        counts_group = QGroupBox("Card Counting"); counts_layout = QGridLayout(); self.count_labels = {}
        for i, name in enumerate(self.counters.keys()):
            counts_layout.addWidget(QLabel(f"{name} RC:"), i, 0); self.count_labels[f"{name}_rc"] = QLabel("0"); counts_layout.addWidget(self.count_labels[f"{name}_rc"], i, 1)
            counts_layout.addWidget(QLabel(f"{name} TC:"), i, 2); self.count_labels[f"{name}_tc"] = QLabel("0.0"); counts_layout.addWidget(self.count_labels[f"{name}_tc"], i, 3)
        counts_group.setLayout(counts_layout); layout.addWidget(counts_group)
        
        shoe_group = QGroupBox("Shoe Status"); shoe_layout = QGridLayout()
        shoe_layout.addWidget(QLabel("Cards Remaining:"), 0, 0); self.cards_remaining_label = QLabel(""); shoe_layout.addWidget(self.cards_remaining_label, 0, 1)
        shoe_layout.addWidget(QLabel("Penetration:"), 1, 0); self.penetration_bar = QProgressBar(); shoe_layout.addWidget(self.penetration_bar, 1, 1)
        shoe_group.setLayout(shoe_layout); layout.addWidget(shoe_group); layout.addStretch()
        return layout

    def _create_strategy_panel(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        advisor_group = QGroupBox("Advanced Decision Advisor (Omega II + Bayes)"); advisor_layout = QVBoxLayout()
        self.advisor_output = QLabel("Enter Player and Dealer cards for advice."); self.advisor_output.setWordWrap(True); self.advisor_output.setStyleSheet("font-size: 14px; color: #336699;")
        advisor_layout.addWidget(self.advisor_output); advisor_group.setLayout(advisor_layout); layout.addWidget(advisor_group)
        
        sim_strat_group = QGroupBox("Simulation & Betting Strategy"); sim_strat_layout = QVBoxLayout()
        self.sim_output = QTextEdit(); self.sim_output.setReadOnly(True); self.sim_output.setFont(QFont("Courier New", 10))
        self.sim_output.setText("Enter cards and run simulation to get betting advice.")
        sim_strat_layout.addWidget(self.sim_output); sim_strat_group.setLayout(sim_strat_layout); layout.addWidget(sim_strat_group)
        return layout

    def _update_counters(self, card_code, direction):
        for counter in self.counters.values():
            if direction == 'add': counter.update(card_code)
            else: counter.undo(card_code)

    def _card_button_clicked(self, card_code: str):
        if self.player_turn_over: QMessageBox.information(self, "Turn Over", "Player turn is complete."); return
        if self.round_card_count == 0: self._set_selection_mode("player")
        elif self.round_card_count == 1: self._set_selection_mode("dealer")
        elif self.round_card_count == 2: self._set_selection_mode("player")
        
        try:
            self.shoe.remove_card(card_code); self._update_counters(card_code, 'add')
            # CORRECTED: This now works for all modes ("player", "dealer", "burned")
            self.selected_cards[self.current_selection_mode][0].append(card_code)
            self.action_history.append({'type': 'card', 'card': card_code, 'mode': self.current_selection_mode})
            self.round_card_count += 1
        except (ValueError, KeyError) as e: QMessageBox.warning(self, "Card Error", str(e)); return
        self.update_displays()
    
    def _stand_action(self):
        self.player_turn_over = True
        self.update_displays()

    def _undo_last_action(self):
        if not self.action_history: return
        last_action = self.action_history.pop()
        try:
            card, mode = last_action['card'], last_action['mode']
            self.shoe.restore_card(card); self._update_counters(card, 'remove')
            # CORRECTED: Now correctly removes the card from the nested list for all modes.
            self.selected_cards[mode][0].remove(card)
            self.round_card_count -= 1
            self.player_turn_over = False
        except (ValueError, KeyError, IndexError) as e:
            QMessageBox.critical(self, "Undo Error", f"Could not undo action: {e}"); self.action_history.append(last_action)
        self.update_displays()

    def _reset_shoe(self):
        if QMessageBox.question(self, "Confirm Reset", "Reset the shoe?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.shoe.reset_shoe(); [c.reset() for c in self.counters.values()]; self._init_state()
            self.sim_output.setText("Enter cards and run simulation to get betting advice.")
            self._set_selection_mode(None); self.update_displays()

    def update_displays(self):
        display_text = f"Player 1  : {' '.join(self.selected_cards['player'][0])}\n"
        display_text += f"Dealer    : {' '.join(self.selected_cards['dealer'][0])}\n"
        display_text += f"Burned    : {' '.join(self.selected_cards['burned'][0])}"
        self.selected_cards_display.setText(display_text)

        decks_rem = self.shoe.decks_remaining()
        for name, counter in self.counters.items():
            rc, tc = counter.running_count, counter.true_count(decks_rem)
            self.count_labels[f"{name}_rc"].setText(f"{rc:g}"); self.count_labels[f"{name}_tc"].setText(f"{tc:.2f}")
        self.cards_remaining_label.setText(f"{self.shoe.total_cards} / {self.shoe.initial_card_count}")
        self.penetration_bar.setValue(int(self.shoe.get_penetration() * 100))
        for code, btn in self.card_buttons.items(): btn.setEnabled(self.shoe.cards.get(code, 0) > 0)

        self.stand_button.setEnabled(bool(self.selected_cards['player'][0]) and not self.player_turn_over)

        player_hand = self.selected_cards["player"][0]
        dealer_card = self.selected_cards["dealer"][0][0] if self.selected_cards["dealer"][0] else None
        if player_hand and dealer_card:
            omega_tc = self.counters["Omega II"].true_count(decks_rem)
            self.advisor_output.setText(decision_advisor.recommend_action(player_hand, dealer_card, omega_tc, self.shoe))
        else:
            self.advisor_output.setText("Enter Player and Dealer cards for advice.")

        if self.last_sim_results:
            recs = self.strategy_advisor.generate_recommendations(self.shoe, self.counters, self.last_sim_results, None)
            self.sim_output.setText("\n".join(recs))

    def _run_simulation(self):
        self.run_sim_button.setEnabled(False); self.run_sim_button.setText("Simulating...")
        thread = QThread(); worker = SimulationWorker(self.shoe.get_remaining_cards())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_simulation_finished); worker.error.connect(self._on_simulation_error)
        worker.finished.connect(thread.quit); worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater); self.thread = thread
        
        self._reset_round_state()

    def _reset_round_state(self):
        self.selected_cards = {"player": [[]], "dealer": [[]], "burned": [[]]}
        self.player_turn_over = False
        self.action_history = []; self.round_card_count = 0
        self.update_displays()

    def _on_simulation_finished(self, results):
        self.last_sim_results = results
        self.run_sim_button.setEnabled(True); self.run_sim_button.setText("Run Simulation")
        self.update_displays()

    def _on_simulation_error(self, err):
        QMessageBox.critical(self, "Simulation Error", err)
        self.run_sim_button.setEnabled(True); self.run_sim_button.setText("Run Simulation")

    def _on_radio_toggled(self, mode: str): self.current_selection_mode = mode
    def _set_selection_mode(self, mode: str | None):
        buttons = {'player': self.radio_player, 'dealer': self.radio_dealer, 'burned': self.radio_burned}
        for b in buttons.values(): b.setAutoExclusive(False); b.setChecked(False); b.setAutoExclusive(True)
        if mode and mode in buttons: buttons[mode].setChecked(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())