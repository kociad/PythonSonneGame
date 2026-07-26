"""Unit tests for advanced AI simulation behavior."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.ai_player import AIPlayer, AIPreset


class _RotatingCardStub:
    """Minimal card-like object with rotation support."""

    def __init__(self, rotation: int = 0):
        self.rotation = rotation

    def rotate(self) -> None:
        self.rotation = (self.rotation + 90) % 360


class _StructureStub:
    """Minimal structure stub for meeple-placement tests."""

    def __init__(self, structure_type: str = "Road", completed: bool = False):
        self._structure_type = structure_type
        self._completed = completed
        self.card_sides = [(MagicMock(get_position=lambda: {"X": 1, "Y": 1}), "N")]

    def get_figures(self):
        return []

    def get_is_completed(self):
        return self._completed

    def get_structure_type(self):
        return self._structure_type


class AIPlayerAdvancedTests(unittest.TestCase):
    """Validate advanced AI decision pipeline for tiles and meeples."""

    def setUp(self) -> None:
        self.ai = AIPlayer("AI_NORMAL_Test", 0, "blue", "NORMAL")

    def test_get_multiple_valid_placements_uses_session_candidates(self):
        """AI should receive all valid placements and align card rotations."""
        game_session = MagicMock()
        original_card = MagicMock()
        game_session.get_valid_placements.return_value = [(1, 2, 90), (3, 4, 180)]

        first_copy = _RotatingCardStub(rotation=0)
        second_copy = _RotatingCardStub(rotation=0)

        with patch.object(
                self.ai,
                "_create_card_copy",
                side_effect=[first_copy, second_copy]) as create_copy:
            placements = self.ai._get_multiple_valid_placements(
                game_session, original_card)

        self.assertEqual(len(placements), 2)
        self.assertEqual(placements[0][:3], (1, 2, 90))
        self.assertEqual(placements[1][:3], (3, 4, 180))
        self.assertEqual(first_copy.rotation, 90)
        self.assertEqual(second_copy.rotation, 180)
        self.assertEqual(create_copy.call_count, 2)

    def test_completion_ratio_counts_connected_structure_edges(self):
        """Completion ratio should measure closed edges, not placed cards."""
        connected_card = MagicMock()
        connected_card.get_neighbors.return_value = {"N": MagicMock()}
        open_card = MagicMock()
        open_card.get_neighbors.return_value = {"S": None}
        structure = MagicMock()
        structure.get_structure_type.return_value = "Road"
        structure.get_is_completed.return_value = False
        structure.card_sides = {
            (connected_card, "N"),
            (open_card, "S"),
        }

        ratio = self.ai._calculate_structure_completion_ratio(structure)

        self.assertEqual(ratio, 0.5)

    def test_candidate_structure_resolves_neighbor_before_tile_is_placed(self):
        """Strategic scoring should see structures beside an empty candidate cell."""
        structure = MagicMock()
        game_session = MagicMock()
        game_session.structure_map = {(4, 3, "S"): structure}

        resolved = self.ai._get_candidate_structure(game_session, 4, 4,
                                                     "N")

        self.assertIs(resolved, structure)

    def test_candidate_monastery_ratio_counts_all_surrounding_cards(self):
        """A prospective monastery should count its eight neighboring cells."""
        board = MagicMock()
        occupied = {(3, 3), (4, 3), (5, 4), (5, 5)}
        board.get_card.side_effect = lambda x, y: (
            MagicMock() if (x, y) in occupied else None)
        game_session = MagicMock()
        game_session.get_game_board.return_value = board

        ratio = self.ai._calculate_candidate_monastery_completion_ratio(
            game_session, 4, 4)

        self.assertEqual(ratio, 0.5)

    def test_card_evaluation_scores_same_structure_only_once(self):
        """Two candidate edges must not duplicate whole-structure bonuses."""
        structure = MagicMock()
        structure.get_is_completed.return_value = False
        structure.get_figures.return_value = [MagicMock()]
        structure.get_structure_type.return_value = "City"
        structure.card_sides = {("card", "N")}
        card = MagicMock()
        card.get_terrains.return_value = {"N": "city", "E": "city"}
        game_session = MagicMock()
        game_session.structure_map = {
            (4, 3, "S"): structure,
            (5, 4, "W"): structure,
        }
        game_session.get_game_board.return_value.get_center.return_value = 4

        with patch.object(
                self.ai,
                "_calculate_structure_completion_ratio",
                return_value=0.0), patch.object(
                    self.ai, "_evaluate_city_specific",
                    return_value=10.0) as city_specific:
            score = self.ai._evaluate_card_placement_advanced(
                game_session, 4, 4, card)

        self.assertEqual(score, self.ai._preset["structure_connection"] + 10.0)
        city_specific.assert_called_once()

    def test_advanced_meeple_placement_can_claim_monastery_center(self):
        """AI should consider the center of a newly placed monastery tile."""
        card = MagicMock()
        card.get_terrains.return_value = {"C": "monastery"}
        monastery = _StructureStub(structure_type="Monastery")
        game_session = MagicMock()
        game_session.last_placed_card = card
        game_session.structure_map = {(4, 4, "C"): monastery}
        game_session.play_figure.return_value = True

        with patch.object(
                self.ai,
                "_evaluate_figure_placement_advanced",
                return_value=60.0), patch.object(
                    self.ai, "_check_and_score_completed_structures"):
            self.ai._handle_figure_placement_advanced(game_session, 4, 4)

        game_session.play_figure.assert_called_once_with(self.ai, 4, 4, "C")
        game_session.next_turn.assert_called_once()

    def test_size_penalties_increase_with_structure_size(self):
        """Every preset should increasingly discourage oversized structures."""
        for preset in (AIPreset.EASY, AIPreset.NORMAL, AIPreset.HARD,
                       AIPreset.EXPERT):
            small, medium, large = preset["size_penalties"]
            self.assertGreater(0, small)
            self.assertGreater(small, medium)
            self.assertGreater(medium, large)

    def test_worker_selects_best_move_from_advanced_simulation_score(self):
        """Worker should choose move with highest simulated advanced score."""
        game_session = MagicMock()
        game_session.get_current_card.return_value = MagicMock()

        candidate_1 = (1, 1, 0, MagicMock())
        candidate_2 = (2, 2, 90, MagicMock())

        self.ai._worker_turn_token = 7
        self.ai._worker_running = True

        with patch.object(
                self.ai,
                "_get_multiple_valid_placements",
                return_value=[candidate_1, candidate_2]), \
                patch.object(self.ai, "_evaluate_card_placement_advanced",
                             return_value=1.0), \
                patch.object(self.ai, "_evaluate_figure_opportunity_advanced",
                             return_value=1.0), \
                patch.object(self.ai, "_evaluate_opponent_blocking",
                             return_value=1.0), \
                patch.object(self.ai, "_evaluate_multi_turn_potential",
                             return_value=1.0), \
                patch.object(
                    self.ai,
                    "_simulate_card_copy_placement_advanced",
                    side_effect=[20.0, 99.0]), \
                patch("models.ai_player.settings_manager.get",
                      return_value=5):
            self.ai._compute_best_move_worker(game_session, 7, (1, 2, 3))

        self.assertIsNotNone(self.ai._worker_result)
        self.assertTrue(self.ai._worker_result["is_valid"])
        self.assertEqual(self.ai._worker_result["best_move"], candidate_2)
        self.assertEqual(self.ai._worker_progress, 1.0)
        self.assertFalse(self.ai._worker_running)

    def test_execute_best_move_places_card_successfully(self):
        """Best-move execution should place card and continue to meeple phase."""
        current_card = _RotatingCardStub(rotation=0)
        game_session = MagicMock()
        game_session.get_current_card.return_value = current_card
        game_session.play_card.return_value = True

        self.ai._ai_thinking_data = {
            "best_move": (5, 6, 180, MagicMock()),
        }

        with patch.object(self.ai, "_handle_figure_placement_advanced") as handle_meeple:
            self.ai._execute_best_move(game_session)

        self.assertEqual(current_card.rotation, 180)
        game_session.play_card.assert_called_once_with(5, 6)
        game_session.set_turn_phase.assert_called_once_with(2)
        handle_meeple.assert_called_once_with(game_session, 5, 6)
        self.assertIsNone(self.ai._ai_thinking_state)
        self.assertIsNone(self.ai._ai_thinking_data)

    def test_handle_figure_placement_advanced_places_best_scored_meeple(self):
        """Advanced meeple logic should choose the highest-scoring open structure."""
        card = MagicMock()
        card.get_terrains.return_value = {
            "N": "road",
            "E": "city",
            "S": "field",
            "W": "field",
        }

        north_structure = _StructureStub(structure_type="Road")
        east_structure = _StructureStub(structure_type="City")

        game_session = MagicMock()
        game_session.last_placed_card = card
        game_session.structure_map = {
            (8, 9, "N"): north_structure,
            (8, 9, "E"): east_structure,
        }
        game_session.play_figure.return_value = True

        with patch.object(self.ai, "_should_conserve_figure",
                          return_value=False), \
                patch.object(
                    self.ai,
                    "_evaluate_figure_placement_advanced",
                    side_effect=lambda _session, _x, _y, direction: {
                        "N": 20.0,
                        "E": 60.0
                    }.get(direction, -5.0),
                ), \
                patch.object(self.ai, "_check_and_score_completed_structures"):
            self.ai._handle_figure_placement_advanced(game_session, 8, 9)

        game_session.play_figure.assert_called_once_with(self.ai, 8, 9, "E")
        game_session.next_turn.assert_called_once()
        game_session.skip_current_action.assert_not_called()


if __name__ == "__main__":
    unittest.main()
