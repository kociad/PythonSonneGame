import logging
import random
import threading
import typing
import time
from typing import List, Tuple, Optional, Dict, Any

from models.player import Player
from models.figure import Figure
from models.card import Card

from utils.settings_manager import settings_manager

logger = logging.getLogger(__name__)


class AIPreset:
    """Configuration presets for different AI difficulty levels."""

    EASY = {
        # "size_bonus": 20,  # Reserved; currently unused by the evaluator.
        "field_multiplier": 0.8,
        "conservation_threshold": 4,
        "placement_threshold": 10000,
        "center_penalty": 2.0,
        "figure_opportunity": 80,
        "structure_connection": 15,
        "unoccupied_bonus": 25,
        "completion_ratio_bonuses": [30, 50, 70, 90],
        "size_penalties": [-15, -35, -60],
        "field_bonuses": [10, 20, 30],
        "city_bonuses": [30, 50, 70],
        "road_bonuses": [25, 40, 55],
        "monastery_bonuses": [60, 90, 120]
    }

    NORMAL = {
        # "size_bonus": 30,  # Reserved; currently unused by the evaluator.
        "field_multiplier": 0.8,
        "conservation_threshold": 3,
        "placement_threshold": 220,
        "center_penalty": 2.2,
        "figure_opportunity": 90,
        "structure_connection": 15,
        "unoccupied_bonus": 25,
        "completion_ratio_bonuses": [25, 50, 75, 100],
        "size_penalties": [-15, -35, -65],
        "field_bonuses": [10, 20, 30],
        "city_bonuses": [30, 50, 70],
        "road_bonuses": [25, 40, 55],
        "monastery_bonuses": [60, 90, 130]
    }

    HARD = {
        # "size_bonus": 30,  # Reserved; currently unused by the evaluator.
        "field_multiplier": 1.0,
        "conservation_threshold": 4,
        "placement_threshold": 210,
        "center_penalty": 1.8,
        "figure_opportunity": 105,
        "structure_connection": 18,
        "unoccupied_bonus": 30,
        "completion_ratio_bonuses": [30, 60, 90, 120],
        "size_penalties": [-20, -45, -85],
        "field_bonuses": [12, 22, 32],
        "city_bonuses": [35, 55, 75],
        "road_bonuses": [28, 42, 58],
        "monastery_bonuses": [70, 105, 145]
    }

    EXPERT = {
        # "size_bonus": 30,  # Reserved; currently unused by the evaluator.
        "field_multiplier": 1.0,
        "conservation_threshold": 2,
        "placement_threshold": 200,
        "center_penalty": 1.0,
        "figure_opportunity": 140,
        "structure_connection": 20,
        "unoccupied_bonus": 30,
        "completion_ratio_bonuses": [30, 70, 130, 200],
        "size_penalties": [-25, -55, -100],
        "field_bonuses": [15, 25, 35],
        "city_bonuses": [35, 70, 110],
        "road_bonuses": [30, 60, 95],
        "monastery_bonuses": [80, 150, 240]
    }


class AIPlayer(Player):
    """
    AI-controlled player with configurable difficulty presets.
    
    The AI uses a preset system to control its behavior and decision-making. Each preset
    contains various parameters that affect how the AI evaluates moves, prioritizes actions,
    and manages resources (meeples).
    
    Preset System:
    - EASY: Conservative play, focuses on simple completions, avoids complex strategies
    - NORMAL: Balanced approach, good mix of completion and expansion
    - HARD: Aggressive play, prioritizes quick completions and meeple liberation
    - EXPERT: Advanced strategies, sophisticated evaluation with multi-turn planning
    
    Key Preset Parameters:
    - size_bonus: Reserved for future use; currently not part of move evaluation
    - field_multiplier: Multiplier for field scoring potential (higher = more field-focused)
    - conservation_threshold: When to conserve meeples (lower = more aggressive placement)
    - placement_threshold: Minimum score to place a meeple (higher = more selective)
    - center_penalty: Penalty per tile of distance from center (higher = stays closer to center)
    - meeple_opportunity: Bonus for meeple placement opportunities (higher = places more meeples)
    - structure_connection: Bonus for connecting to existing structures (higher = more connection-focused)
    - unoccupied_bonus: Bonus for unoccupied structures (higher = prefers empty structures)
    
    Structure-Specific Bonuses (arrays for different structure sizes):
    - road_bonuses: Points for roads of different sizes [small, medium, large]
    - city_bonuses: Points for cities of different sizes [small, medium, large]
    - monastery_bonuses: Points for monasteries based on completion proximity [near, mid, far]
    - field_bonuses: Points for field control and potential [low, medium, high]
    
    Advanced Parameters:
    - size_penalties: Penalties for structures that tie up meeples [small, medium, large]
    - completion_ratio_bonuses: Bonuses based on completion percentage [25%, 50%, 75%, 100%]
    
    The AI adapts its strategy based on game phase (early/mid/late) and uses these
    preset values to evaluate card placements, meeple placements, and overall game strategy.
    """

    def __init__(self,
                 name: str,
                 index: int,
                 color: str,
                 difficulty: str = "NORMAL") -> None:
        """
        Initialize an AI player.
        
        Args:
            name: The player's name
            index: The player's index in the game
            color: The player's color
            difficulty: AI difficulty level (EASY, NORMAL, HARD, EXPERT)
        """
        super().__init__(name, color, index, is_ai=True)
        self._game_phase = "early"
        self._last_score = 0
        self._consecutive_low_scores = 0
        self._difficulty = difficulty.upper()
        self._preset = self._get_preset()

        self._ai_thinking_state = None
        self._ai_thinking_progress = 0
        self._ai_thinking_start_time = None

        self._evaluation_cache = {}
        self._evaluation_cache_valid = False
        self._last_board_state = None

        self._figure_cache = {}
        self._figure_cache_valid = False
        self._cache_lock = threading.Lock()
        self._worker_cache_context = threading.local()

        self._worker_thread = None
        self._worker_lock = threading.Lock()
        self._worker_result = None
        self._worker_progress = 0.0
        self._worker_running = False
        self._worker_turn_token = 0

    def _get_preset(self) -> Dict[str, Any]:
        """Get the AI preset configuration for the current difficulty."""
        if self._difficulty == "EASY":
            return AIPreset.EASY
        elif self._difficulty == "HARD":
            return AIPreset.HARD
        elif self._difficulty == "EXPERT":
            return AIPreset.EXPERT
        else:
            return AIPreset.NORMAL

    def play_turn(self, game_session: 'GameSession') -> None:
        """
        Perform the AI's turn logic with configurable strategy.
        
        The AI will either use simple placement logic or advanced simulation
        based on the AI_USE_SIMULATION setting.
        
        Args:
            game_session: The current game session
        """
        if not settings_manager.get("AI_USE_SIMULATION", False):
            if self._ai_thinking_state is not None:
                self._continue_thinking(game_session)
                return
            self._play_turn_simple(game_session)
            return

        with self._worker_lock:
            worker_running = self._worker_running
            worker_result = self._worker_result
            worker_token = self._worker_turn_token
            current_turn_state = game_session.get_turn_state_token()

        if worker_running:
            return

        if worker_result and worker_result["turn_token"] == worker_token:
            if worker_result.get("turn_state") != current_turn_state:
                logger.debug(
                    "Discarding stale AI worker result due to turn state mismatch: expected=%s, got=%s",
                    current_turn_state,
                    worker_result.get("turn_state"),
                )
                self._clear_worker_state()
            elif worker_result.get("is_valid"):
                self._ai_thinking_data = {"best_move": worker_result["best_move"]}
                self._execute_best_move(game_session)
                self._clear_worker_state()
                return
            else:
                self._clear_worker_state()
                return

        logger.info(f"Player {self.name} is thinking...")
        self._update_game_phase(game_session)
        self._invalidate_evaluation_cache()
        self._invalidate_figure_cache()

        turn_state = game_session.get_turn_state_token()

        with self._worker_lock:
            self._worker_turn_token += 1
            turn_token = self._worker_turn_token
            self._worker_running = True
            self._worker_progress = 0.0
            self._worker_result = None
            self._worker_thread = threading.Thread(
                target=self._compute_best_move_worker,
                args=(game_session, turn_token, turn_state),
                daemon=True)
            self._worker_thread.start()
        return

    def _clear_worker_state(self) -> None:
        """Reset worker state after completing or consuming a result."""
        with self._worker_lock:
            self._worker_thread = None
            self._worker_result = None
            self._worker_progress = 0.0
            self._worker_running = False

    def _compute_best_move_worker(
        self,
        game_session: 'GameSession',
        turn_token: int,
        turn_state: tuple[int, int, Optional[int]],
    ) -> None:
        """Compute the best move in a background worker thread."""
        result = {
            "turn_token": turn_token,
            "turn_state": turn_state,
            "is_valid": False,
            "best_move": None
        }

        try:
            self._worker_cache_context.evaluation_cache = {}
            self._worker_cache_context.figure_cache = {}

            current_card = game_session.get_current_card()
            possible_placements = self._get_multiple_valid_placements(
                game_session, current_card)

            if possible_placements:
                strategic_scores = []
                total_placements = len(possible_placements)
                for idx, placement in enumerate(possible_placements, start=1):
                    x, y, rotations_needed, card_copy = placement

                    strategic_score = self._evaluate_card_placement_advanced(
                        game_session, x, y, card_copy)
                    strategic_score += self._evaluate_figure_opportunity_advanced(
                        game_session, x, y, card_copy)
                    strategic_score += self._evaluate_multi_turn_potential(
                        game_session, x, y, card_copy)
                    strategic_scores.append((strategic_score, placement))

                    with self._worker_lock:
                        self._worker_progress = (idx / total_placements) * 0.5

                strategic_scores.sort(reverse=True, key=lambda x: x[0])
                max_candidates = settings_manager.get(
                    "AI_STRATEGIC_CANDIDATES", 5)
                top_candidates = strategic_scores if max_candidates == -1 else strategic_scores[
                    :max_candidates]

                best_move = None
                best_score = float("-inf")
                total_candidates = max(1, len(top_candidates))
                for idx, (_, placement) in enumerate(top_candidates, start=1):
                    x, y, rotations_needed, card_copy = placement
                    card_score = self._simulate_card_copy_placement_advanced(
                        game_session, x, y, card_copy)
                    if card_score > best_score:
                        best_score = card_score
                        best_move = placement

                    with self._worker_lock:
                        self._worker_progress = 0.5 + (
                            idx / total_candidates) * 0.5

                result["is_valid"] = True
                result["best_move"] = best_move
            else:
                result["is_valid"] = True
        finally:
            self._worker_cache_context.evaluation_cache = None
            self._worker_cache_context.figure_cache = None

            with self._worker_lock:
                if turn_token == self._worker_turn_token:
                    self._worker_progress = 1.0
                    self._worker_result = result
                    self._worker_running = False
        return

    def _update_game_phase(self, game_session: 'GameSession') -> None:
        """Update the current game phase based on cards played."""
        total_cards = len(game_session.get_cards_deck()) + 1
        cards_played = 71 - total_cards

        if cards_played < 15:
            self._game_phase = "early"
        elif cards_played < 45:
            self._game_phase = "mid"
        else:
            self._game_phase = "late"

    def _play_turn_simple(self, game_session: 'GameSession') -> None:
        """
        Execute simple AI logic without simulation.
        
        Uses random valid placement and basic meeple placement logic.
        
        Args:
            game_session: The current game session
        """
        current_card = game_session.get_current_card()
        placement = game_session.get_random_valid_placement(current_card)
        if not placement:
            logger.info(
                f"Player {self.name} couldn't place their card anywhere and discarded it"
            )
            game_session.skip_current_action()
            return

        x, y, rotations_needed = placement
        for _ in range(rotations_needed):
            current_card.rotate()

        if game_session.play_card(x, y):
            game_session.set_turn_phase(2)
            self._handle_figure_placement_simple(game_session, x, y)
        else:
            logger.error(
                f"Player {self.name} failed to place card at validated position [{x},{y}]"
            )
            game_session.skip_current_action()

    def _start_advanced_thinking(self, game_session: 'GameSession') -> None:
        """Start the progressive AI thinking process."""
        self._ai_thinking_state = "evaluating_placements"
        self._ai_thinking_progress = 0
        self._ai_thinking_start_time = time.time()

        self._invalidate_evaluation_cache()
        self._invalidate_figure_cache()
        self._ai_thinking_data = {
            'game_session': game_session,
            'current_card': game_session.get_current_card(),
            'possible_placements': None,
            'strategic_scores': [],
            'top_candidates': [],
            'best_move': None,
            'best_score': float('-inf')
        }

        self._ai_thinking_data[
            'possible_placements'] = self._get_multiple_valid_placements(
                game_session, self._ai_thinking_data['current_card'])

        if not self._ai_thinking_data['possible_placements']:
            logger.info(
                f"Player {self.name} couldn't place their card anywhere and discarded it"
            )
            game_session.skip_current_action()
            self._ai_thinking_state = None
            return

    def _continue_thinking(self, game_session: 'GameSession') -> None:
        """
        Continue the progressive AI thinking process.
        
        This method implements a time-based yielding system that allows the AI to think
        progressively while keeping the UI responsive.
        
        **Time-Based Yielding Process:**
        
        1. AI Thinking Phase: AI processes card placements, evaluations, and simulations
        2. Time Limit Check: When AI_THINKING_SPEED time is reached, AI yields control to UI
        3. Timer Reset: Timer resets to 0 for the next thinking step
        4. UI Responsiveness: UI gets control back, renders smoothly
        5. Continue Thinking: AI continues from where it left off in the next frame
             
        Settings:
        - Fast (0.1s): Maximum UI responsiveness, slower AI completion
        - Balanced (0.5s): Good balance of AI speed and UI responsiveness  
        - Slow (2.0s): Faster AI completion, less UI responsiveness
        - Unlimited (-1): AI completes each phase entirely, may cause brief UI pauses
        
        Args:
            game_session: The current game session
        """
        current_time = time.time()
        max_thinking_time = self._get_ai_thinking_max_time()

        if (max_thinking_time != -1
                and current_time - self._ai_thinking_start_time > max_thinking_time):
            self._ai_thinking_start_time = current_time
            return

        if self._ai_thinking_state == "evaluating_placements":
            self._continue_evaluating_placements()
        elif self._ai_thinking_state == "simulating_candidates":
            self._continue_simulating_candidates()
        elif self._ai_thinking_state == "executing_move":
            self._execute_best_move(game_session)

    def _get_ai_thinking_max_time(self) -> float:
        """Get AI thinking step duration from settings (-1 means unlimited)."""
        configured_speed = settings_manager.get("AI_THINKING_SPEED", 0.5)
        try:
            max_time = float(configured_speed)
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid AI_THINKING_SPEED value '{configured_speed}', falling back to 0.5"
            )
            return 0.5

        if max_time <= -1:
            return -1
        return max_time

    def _continue_evaluating_placements(self) -> None:
        """Continue evaluating strategic placements."""
        data = self._ai_thinking_data
        possible_placements = data['possible_placements']
        strategic_scores = data['strategic_scores']

        placements_per_step = 3
        start_idx = self._ai_thinking_progress
        end_idx = min(start_idx + placements_per_step,
                      len(possible_placements))

        for i in range(start_idx, end_idx):
            placement = possible_placements[i]
            x, y, rotations_needed, card_copy = placement

            strategic_score = self._evaluate_card_placement_advanced(
                data['game_session'], x, y, card_copy)
            strategic_score += self._evaluate_figure_opportunity_advanced(
                data['game_session'], x, y, card_copy)
            strategic_score += self._evaluate_multi_turn_potential(
                data['game_session'], x, y, card_copy)

            strategic_scores.append((strategic_score, placement))

        self._ai_thinking_progress = end_idx

        if end_idx >= len(possible_placements):
            strategic_scores.sort(reverse=True, key=lambda x: x[0])
            max_candidates = settings_manager.get("AI_STRATEGIC_CANDIDATES", 5)
            if max_candidates == -1:
                data['top_candidates'] = strategic_scores
            else:
                data['top_candidates'] = strategic_scores[:max_candidates]

            self._ai_thinking_state = "simulating_candidates"
            self._ai_thinking_progress = 0

    def _continue_simulating_candidates(self) -> None:
        """Continue simulating top candidates."""
        data = self._ai_thinking_data
        top_candidates = data['top_candidates']

        if self._ai_thinking_progress < len(top_candidates):
            strategic_score, placement = top_candidates[
                self._ai_thinking_progress]
            x, y, rotations_needed, card_copy = placement

            card_score = self._simulate_card_placement_advanced(
                data['game_session'], x, y, rotations_needed)

            if card_score > data['best_score']:
                data['best_score'] = card_score
                data['best_move'] = placement

            self._ai_thinking_progress += 1
        else:
            self._ai_thinking_state = "executing_move"

    def _execute_best_move(self, game_session: 'GameSession') -> None:
        """Execute the best move found during thinking."""
        data = self._ai_thinking_data
        best_move = data['best_move']

        if best_move:
            x, y, rotations_needed, card_copy = best_move
            current_card = game_session.get_current_card()

            original_rotation = current_card.rotation
            while current_card.rotation != rotations_needed:
                current_card.rotate()

            if game_session.play_card(x, y):
                game_session.set_turn_phase(2)
                self._handle_figure_placement_advanced(game_session, x, y)
            else:
                logger.error(
                    f"Player {self.name} failed to place card at validated position [{x},{y}]"
                )
                while current_card.rotation != original_rotation:
                    current_card.rotate()
                game_session.skip_current_action()
        else:
            logger.info(
                f"Player {self.name} couldn't find any valid placements and will discard the card"
            )
            game_session.skip_current_action()

        self._ai_thinking_state = None
        self._ai_thinking_data = None

    def is_thinking(self) -> bool:
        """Check if the AI is currently in a thinking state."""
        if self._ai_thinking_state is not None:
            return True

        with self._worker_lock:
            return self._worker_running or self._worker_result is not None

    def get_thinking_progress(self) -> float:
        """Get the current thinking progress as a percentage (0.0 to 1.0)."""
        with self._worker_lock:
            if self._worker_running:
                return min(max(self._worker_progress, 0.0), 1.0)
            if self._worker_result is not None:
                return 1.0

        if not self.is_thinking():
            return 0.0

        if self._ai_thinking_state == "evaluating_placements":
            data = self._ai_thinking_data
            total_placements = max(1, len(data['possible_placements']))
            return self._ai_thinking_progress / total_placements * 0.5  # First 50%
        elif self._ai_thinking_state == "simulating_candidates":
            data = self._ai_thinking_data
            total_candidates = max(1, len(data['top_candidates']))
            progress = self._ai_thinking_progress / total_candidates
            return 0.5 + progress * 0.5  # Last 50%
        else:
            return 1.0

    def _get_evaluation_cache_key(self, card: Card, x: int, y: int,
                                  evaluation_type: str) -> tuple:
        """Get a cache key for AI evaluation."""
        return (id(card), x, y, card.rotation, evaluation_type)

    def _invalidate_evaluation_cache(self) -> None:
        """Invalidate the AI evaluation cache."""
        with self._cache_lock:
            self._evaluation_cache.clear()
            self._evaluation_cache_valid = False
            self._last_board_state = None

    def invalidate_evaluation_cache(self) -> None:
        """Public method to invalidate the AI evaluation cache."""
        self._invalidate_evaluation_cache()

    def _get_figure_cache_key(self, x: int, y: int, direction: str,
                              figure_type: str) -> tuple:
        """Get a cache key for figure placement evaluation."""
        return (x, y, direction, figure_type)

    def _invalidate_figure_cache(self) -> None:
        """Invalidate the figure placement cache."""
        with self._cache_lock:
            self._figure_cache.clear()
            self._figure_cache_valid = False

    def invalidate_figure_cache(self) -> None:
        """Public method to invalidate the figure placement cache."""
        self._invalidate_figure_cache()

    def _evaluate_cached(self, card: Card, x: int, y: int,
                         evaluation_type: str, evaluation_func) -> float:
        """Evaluate with caching support."""
        cache_key = self._get_evaluation_cache_key(card, x, y, evaluation_type)
        local_cache = getattr(self._worker_cache_context, "evaluation_cache", None)

        if local_cache is not None:
            if cache_key in local_cache:
                return local_cache[cache_key]

            result = evaluation_func()
            local_cache[cache_key] = result
            return result

        with self._cache_lock:
            if cache_key in self._evaluation_cache:
                return self._evaluation_cache[cache_key]

        result = evaluation_func()

        with self._cache_lock:
            self._evaluation_cache[cache_key] = result

        return result

    def _evaluate_figure_cached(self, x: int, y: int, direction: str,
                                figure_type: str, evaluation_func) -> float:
        """Evaluate figure placement with caching support."""
        cache_key = self._get_figure_cache_key(x, y, direction, figure_type)
        local_cache = getattr(self._worker_cache_context, "figure_cache", None)

        if local_cache is not None:
            if cache_key in local_cache:
                return local_cache[cache_key]

            result = evaluation_func()
            local_cache[cache_key] = result
            return result

        with self._cache_lock:
            if cache_key in self._figure_cache:
                return self._figure_cache[cache_key]

        result = evaluation_func()

        with self._cache_lock:
            self._figure_cache[cache_key] = result

        return result

    def _get_multiple_valid_placements(
            self, game_session: 'GameSession',
            card: Card) -> List[Tuple[int, int, int, Card]]:
        """
        Get all valid card placements using the game's existing validation.
        
        Args:
            game_session: The current game session
            card: The card to find placements for
            
        Returns:
            list of tuples containing (x, y, rotation, card_copy) for each valid placement
        """
        placements = []

        valid_placements = game_session.get_valid_placements(card)
        logger.debug(
            f"AI {self.name} found {len(valid_placements)} valid placements")

        for x, y, card_rotation in valid_placements:
            card_copy = self._create_card_copy(card)
            while card_copy.rotation != card_rotation:
                card_copy.rotate()
            placements.append((x, y, card_rotation, card_copy))
            logger.debug(
                f"AI {self.name} found valid placement at ({x},{y}) with rotation {card_rotation * 90}"
            )

        logger.debug(
            f"AI {self.name} found {len(placements)} valid placements")
        return placements

    def _create_card_copy(self, card: Card) -> Card:
        """
        Create a copy of a card for simulation.
        
        Args:
            card: The card to copy
            
        Returns:
            A new Card instance with the same properties
        """
        card_copy = Card(
            image_path=card.image_path,
            terrains=card.terrains.copy(),
            connections=card.connections.copy() if card.connections else None,
            features=card.features.copy() if card.features else None,
            is_starting_card=card.get_is_starting_card())
        card_copy.rotation = card.rotation
        return card_copy

    def _simulate_card_placement_advanced(self, game_session: 'GameSession',
                                          x: int, y: int,
                                          rotations_needed: int) -> float:
        """
        Simulate card placement using advanced strategic evaluation.
        
        Evaluates the potential score of placing a card at the given position
        considering structure completion, field potential, meeple opportunities,
        opponent blocking, and multi-turn potential.
        
        Args:
            game_session: The current game session
            x: X coordinate for placement
            y: Y coordinate for placement
            rotations_needed: Number of rotations needed for the card
            
        Returns:
            A score representing the desirability of this placement
        """
        current_card = game_session.get_current_card()
        original_rotation = current_card.rotation

        for _ in range(rotations_needed):
            current_card.rotate()

        score = self._evaluate_cached(
            current_card, x, y, "placement",
            lambda: self._evaluate_card_placement_advanced(
                game_session, x, y, current_card))
        score += self._evaluate_cached(
            current_card, x, y, "figure",
            lambda: self._evaluate_figure_opportunity_advanced(
                game_session, x, y, current_card))
        score += self._evaluate_cached(
            current_card, x, y, "structure",
            lambda: self._evaluate_structure_completion_potential(
                game_session, x, y, current_card))
        score += self._evaluate_cached(
            current_card, x, y,
            "field", lambda: self._evaluate_field_potential(
                game_session, x, y, current_card))
        score += self._evaluate_cached(
            current_card, x, y, "multiturn",
            lambda: self._evaluate_multi_turn_potential(
                game_session, x, y, current_card))

        while current_card.rotation != original_rotation:
            current_card.rotate()

        return score

    def _simulate_card_copy_placement_advanced(self, game_session: 'GameSession',
                                               x: int, y: int,
                                               card_copy: Card) -> float:
        """
        Simulate card placement using an already-rotated card copy.

        This helper is safe for threaded worker evaluation because it avoids
        mutating or reading the current live card from ``game_session``.

        Args:
            game_session: The current game session
            x: X coordinate for placement
            y: Y coordinate for placement
            card_copy: Pre-rotated card copy for this candidate

        Returns:
            A score representing the desirability of this placement
        """
        score = self._evaluate_cached(
            card_copy, x, y, "placement",
            lambda: self._evaluate_card_placement_advanced(
                game_session, x, y, card_copy))
        score += self._evaluate_cached(
            card_copy, x, y, "figure",
            lambda: self._evaluate_figure_opportunity_advanced(
                game_session, x, y, card_copy))
        score += self._evaluate_cached(
            card_copy, x, y, "structure",
            lambda: self._evaluate_structure_completion_potential(
                game_session, x, y, card_copy))
        score += self._evaluate_cached(
            card_copy, x, y,
            "field", lambda: self._evaluate_field_potential(
                game_session, x, y, card_copy))
        score += self._evaluate_cached(
            card_copy, x, y, "multiturn",
            lambda: self._evaluate_multi_turn_potential(
                game_session, x, y, card_copy))
        return score

    @staticmethod
    def _get_candidate_structure(game_session: 'GameSession', x: int, y: int,
                                 direction: str) -> Optional['Structure']:
        """Resolve a structure both before and after a candidate tile is placed."""
        structure = game_session.structure_map.get((x, y, direction))
        if structure is not None:
            return structure

        neighbors = {
            "N": (x, y - 1, "S"),
            "E": (x + 1, y, "W"),
            "S": (x, y + 1, "N"),
            "W": (x - 1, y, "E"),
        }
        neighbor_key = neighbors.get(direction)
        return (game_session.structure_map.get(neighbor_key)
                if neighbor_key else None)

    @staticmethod
    def _calculate_candidate_monastery_completion_ratio(
            game_session: 'GameSession', x: int, y: int) -> float:
        """Return the occupied share of the eight cells around a candidate."""
        board = game_session.get_game_board()
        neighbor_offsets = (
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),           (1, 0),
            (-1, 1),  (0, 1),  (1, 1),
        )
        occupied = sum(
            board.get_card(x + offset_x, y + offset_y) is not None
            for offset_x, offset_y in neighbor_offsets)
        return occupied / len(neighbor_offsets)

    @staticmethod
    def _calculate_structure_completion_ratio(structure: 'Structure') -> float:
        """Return the share of a structure's required neighboring tiles present."""
        structure_type = structure.get_structure_type()
        if structure.get_is_completed():
            return 1.0
        if structure_type == "Field":
            return 0.0

        if structure_type == "Monastery":
            center_cards = [
                card for card, direction in structure.card_sides
                if direction == "C"
            ]
            if not center_cards:
                return 0.0

            neighbors = center_cards[0].get_neighbors()
            occupied_positions = {
                direction for direction in ("N", "E", "S", "W")
                if neighbors.get(direction)
            }
            diagonal_directions = {
                "N": (("E", "NE"), ("W", "NW")),
                "E": (("N", "NE"), ("S", "SE")),
                "S": (("E", "SE"), ("W", "SW")),
                "W": (("N", "NW"), ("S", "SW")),
            }
            for direction, diagonals in diagonal_directions.items():
                neighbor = neighbors.get(direction)
                if neighbor:
                    occupied_positions.update(
                        position for neighbor_direction, position in diagonals
                        if neighbor.get_neighbor(neighbor_direction))
            return len(occupied_positions) / 8.0

        openable_sides = [(card, direction)
                          for card, direction in structure.card_sides
                          if direction != "C"]
        if not openable_sides:
            return 0.0
        connected_sides = sum(
            bool(card.get_neighbors().get(direction))
            for card, direction in openable_sides)
        return connected_sides / len(openable_sides)

    def _evaluate_card_placement_advanced(self, game_session: 'GameSession',
                                          x: int, y: int,
                                          card_copy: Card) -> float:
        """
        Evaluate the score of a card placement position using preset configuration.
        
        Args:
            game_session: The current game session
            x: X coordinate for placement
            y: Y coordinate for placement
            card_copy: A copy of the card being evaluated
            
        Returns:
            A score representing the desirability of this placement
        """
        score = 0.0
        terrains = card_copy.get_terrains()
        evaluated_structures = set()

        for direction, terrain_type in terrains.items():
            if direction == "C" and terrains.get(direction) != "monastery":
                continue

            structure = self._get_candidate_structure(game_session, x, y, direction)
            if structure in evaluated_structures:
                continue
            if structure:
                evaluated_structures.add(structure)
                score += self._preset["structure_connection"]

                if not structure.get_figures():
                    score += self._preset["unoccupied_bonus"]

                total_sides = len(structure.card_sides)
                completion_ratio = self._calculate_structure_completion_ratio(structure)

                score += self._completion_ratio_bonus(completion_ratio)

                # Apply size penalties
                if total_sides > 8:
                    score += self._preset["size_penalties"][2]
                elif total_sides > 6:
                    score += self._preset["size_penalties"][1]
                elif total_sides > 4:
                    score += self._preset["size_penalties"][0]

                structure_type = structure.get_structure_type()
                if structure_type == "City":
                    score += self._evaluate_city_specific(
                        game_session, structure, completion_ratio)
                elif structure_type == "Road":
                    score += self._evaluate_road_specific(
                        game_session, structure, completion_ratio)
                elif structure_type == "Monastery":
                    score += self._evaluate_monastery_specific(
                        game_session, structure, completion_ratio)
            elif direction == "C" and terrain_type == "monastery":
                completion_ratio = self._calculate_candidate_monastery_completion_ratio(
                    game_session, x, y)
                score += self._preset["unoccupied_bonus"]
                score += self._completion_ratio_bonus(completion_ratio)
                score += self._evaluate_monastery_specific(
                    game_session, None, completion_ratio)

        center = game_session.get_game_board().get_center()
        distance_from_center = abs(x - center) + abs(y - center)
        score -= distance_from_center * self._preset["center_penalty"]

        for direction, terrain_type in terrains.items():
            if direction == "C" and terrains.get(direction) != "monastery":
                continue
            if not self._get_candidate_structure(game_session, x, y, direction):
                score += 10.0

        return score

    def _completion_ratio_bonus(self, completion_ratio: float) -> float:
        """Select the configured bonus for a structure's completion share."""
        if completion_ratio > 0.9:
            return self._preset["completion_ratio_bonuses"][3]
        if completion_ratio > 0.7:
            return self._preset["completion_ratio_bonuses"][2]
        if completion_ratio > 0.5:
            return self._preset["completion_ratio_bonuses"][1]
        if completion_ratio > 0.3:
            return self._preset["completion_ratio_bonuses"][0]
        return 0.0

    def _evaluate_city_specific(self, game_session: 'GameSession',
                                structure: 'Structure',
                                completion_ratio: float) -> float:
        """Evaluate city-specific scoring potential using preset configuration."""
        score = 0.0

        total_sides = len(structure.card_sides)

        if completion_ratio > 0.8:
            score += self._preset["city_bonuses"][2]
        elif completion_ratio > 0.6:
            score += self._preset["city_bonuses"][1]
        elif completion_ratio > 0.4:
            score += self._preset["city_bonuses"][0]

        if total_sides <= 4:
            score += 60.0
        elif total_sides <= 6:
            score += 40.0
        elif total_sides <= 8:
            score += 25.0

        return score

    def _evaluate_road_specific(self, game_session: 'GameSession',
                                structure: 'Structure',
                                completion_ratio: float) -> float:
        """Evaluate road-specific scoring potential using preset configuration."""
        score = 0.0

        total_sides = len(structure.card_sides)

        if completion_ratio > 0.8:
            score += self._preset["road_bonuses"][2]
        elif completion_ratio > 0.6:
            score += self._preset["road_bonuses"][1]
        elif completion_ratio > 0.4:
            score += self._preset["road_bonuses"][0]

        if total_sides <= 3:
            score += 50.0
        elif total_sides <= 5:
            score += 35.0
        elif total_sides <= 7:
            score += 20.0

        return score

    def _evaluate_monastery_specific(self, game_session: 'GameSession',
                                     structure: Optional['Structure'],
                                     completion_ratio: float) -> float:
        """Evaluate monastery-specific scoring potential using preset configuration."""
        score = 0.0

        if completion_ratio > 0.8:
            score += self._preset["monastery_bonuses"][2]
        elif completion_ratio > 0.6:
            score += self._preset["monastery_bonuses"][1]
        elif completion_ratio > 0.4:
            score += self._preset["monastery_bonuses"][0]

        return score

    def _evaluate_figure_opportunity_advanced(self,
                                              game_session: 'GameSession',
                                              x: int, y: int,
                                              card_copy: Card) -> float:
        """
        Evaluate potential meeple placement opportunities using preset configuration.
        
        Args:
            game_session: The current game session
            x: X coordinate for placement
            y: Y coordinate for placement
            card_copy: A copy of the card being evaluated
            
        Returns:
            A score representing the potential for meeple placement
        """
        score = 0.0
        terrains = card_copy.get_terrains()
        evaluated_structures = set()

        for direction, terrain_type in terrains.items():
            if direction == "C" and terrains.get(direction) != "monastery":
                continue

            structure = self._get_candidate_structure(game_session, x, y, direction)
            if structure in evaluated_structures:
                continue
            if (structure and not structure.get_figures()
                    and not structure.get_is_completed()):
                evaluated_structures.add(structure)
                completion_ratio = self._calculate_structure_completion_ratio(structure)
                score += completion_ratio * self._preset[
                    "figure_opportunity"]

                if len(self.figures) > 0:
                    score += 40.0
                else:
                    score += 5.0

                structure_type = structure.get_structure_type()
                if structure_type == "Field":
                    score += self._evaluate_field_figure_opportunity(
                        game_session, structure)
            elif direction == "C" and terrain_type == "monastery":
                completion_ratio = self._calculate_candidate_monastery_completion_ratio(
                    game_session, x, y)
                score += completion_ratio * self._preset["figure_opportunity"]
                score += 40.0 if len(self.figures) > 0 else 5.0

        return score

    def _evaluate_field_figure_opportunity(self, game_session: 'GameSession',
                                           structure: 'Structure') -> float:
        """Evaluate field meeple placement opportunities using preset configuration."""
        score = 0.0

        completed_cities = [
            s for s in game_session.structures
            if s.get_structure_type() == "City" and s.get_is_completed()
        ]

        touched_cities = set()
        for card, _ in structure.card_sides:
            neighbors = card.get_neighbors().values()
            touched_cards = set([card])
            touched_cards.update([n for n in neighbors if n])

            for city_structure in completed_cities:
                for city_card, _ in city_structure.card_sides:
                    if city_card in touched_cards:
                        touched_cities.add(city_structure)
                        break

        score += len(touched_cities) * 8.0

        field_size = len(structure.card_sides)
        if field_size > 8:
            score += self._preset["field_bonuses"][2]
        elif field_size > 6:
            score += self._preset["field_bonuses"][1]
        elif field_size > 4:
            score += self._preset["field_bonuses"][0]

        return score

    def _evaluate_multi_turn_potential(self, game_session: 'GameSession',
                                       x: int, y: int,
                                       card_copy: Card) -> float:
        """
        Evaluate the potential for future turns and strategic positioning.
        
        Args:
            game_session: The current game session
            x: X coordinate for placement
            y: Y coordinate for placement
            card_copy: A copy of the card being evaluated
            
        Returns:
            A score representing multi-turn potential
        """
        score = 0.0

        if self._game_phase == "early":
            score += self._evaluate_early_game_positioning(
                game_session, x, y, card_copy)
        elif self._game_phase == "mid":
            score += self._evaluate_mid_game_positioning(
                game_session, x, y, card_copy)
        else:
            score += self._evaluate_late_game_positioning(
                game_session, x, y, card_copy)

        return score

    def _evaluate_early_game_positioning(self, game_session: 'GameSession',
                                         x: int, y: int,
                                         card_copy: Card) -> float:
        """Evaluate positioning for early game strategy."""
        score = 0.0

        center = game_session.get_game_board().get_center()
        distance_from_center = abs(x - center) + abs(y - center)

        if distance_from_center <= 2:
            score += 35.0
        elif distance_from_center <= 4:
            score += 20.0
        elif distance_from_center <= 6:
            score += 10.0

        return score

    def _evaluate_mid_game_positioning(self, game_session: 'GameSession',
                                       x: int, y: int,
                                       card_copy: Card) -> float:
        """Evaluate positioning for mid game strategy."""
        score = 0.0

        terrains = card_copy.get_terrains()
        evaluated_structures = set()
        for direction, terrain_type in terrains.items():
            if direction == "C" and terrains.get(direction) != "monastery":
                continue

            structure = self._get_candidate_structure(game_session, x, y, direction)
            if structure in evaluated_structures:
                continue
            if structure and len(structure.card_sides) > 2:
                evaluated_structures.add(structure)
                score += 25.0

        return score

    def _evaluate_late_game_positioning(self, game_session: 'GameSession',
                                        x: int, y: int,
                                        card_copy: Card) -> float:
        """Evaluate positioning for late game strategy."""
        score = 0.0

        terrains = card_copy.get_terrains()
        evaluated_structures = set()
        for direction, terrain_type in terrains.items():
            if direction == "C" and terrains.get(direction) != "monastery":
                continue

            structure = self._get_candidate_structure(game_session, x, y, direction)
            if structure in evaluated_structures:
                continue
            if structure:
                evaluated_structures.add(structure)
                completion_ratio = self._calculate_structure_completion_ratio(structure)

                if completion_ratio > 0.8:
                    score += 70.0
                elif completion_ratio > 0.6:
                    score += 40.0

        return score

    def _evaluate_structure_completion_potential(self,
                                                 game_session: 'GameSession',
                                                 x: int, y: int,
                                                 card: Card) -> float:
        """
        Evaluate potential score gains from structure completion.
        
        Args:
            game_session: The current game session
            x: X coordinate for placement
            y: Y coordinate for placement
            card: The card being evaluated
            
        Returns:
            A score representing the potential for structure completion
        """
        score = 0.0
        terrains = card.get_terrains()
        evaluated_structures = set()

        for direction, terrain_type in terrains.items():
            if direction == "C" and terrains.get(direction) != "monastery":
                continue

            structure = self._get_candidate_structure(game_session, x, y, direction)
            if structure in evaluated_structures:
                continue
            if structure:
                evaluated_structures.add(structure)
                completion_ratio = self._calculate_structure_completion_ratio(structure)

                if completion_ratio > 0.8:
                    score += 100.0
                elif completion_ratio > 0.6:
                    score += 70.0
                elif completion_ratio > 0.4:
                    score += 45.0
                elif completion_ratio > 0.2:
                    score += 25.0

                if not structure.get_figures():
                    score += 20.0
            elif direction == "C" and terrain_type == "monastery":
                completion_ratio = self._calculate_candidate_monastery_completion_ratio(
                    game_session, x, y)
                if completion_ratio > 0.8:
                    score += 100.0
                elif completion_ratio > 0.6:
                    score += 70.0
                elif completion_ratio > 0.4:
                    score += 45.0
                elif completion_ratio > 0.2:
                    score += 25.0
                score += 20.0

        return score

    def _evaluate_field_potential(self, game_session: 'GameSession', x: int,
                                  y: int, card: Card) -> float:
        """
        Evaluate potential score gains from field placement.
        
        Args:
            game_session: The current game session
            x: X coordinate for placement
            y: Y coordinate for placement
            card: The card being evaluated
            
        Returns:
            A score representing the potential for field scoring
        """
        score = 0.0
        terrains = card.get_terrains()
        evaluated_structures = set()

        for direction, terrain_type in terrains.items():
            if direction == "C" and terrains.get(direction) != "monastery":
                continue

            if terrain_type == "field":
                structure = self._get_candidate_structure(game_session, x, y, direction)
                if structure in evaluated_structures:
                    continue
                if structure:
                    evaluated_structures.add(structure)
                    completed_cities = [
                        s for s in game_session.structures
                        if s.get_structure_type() == "City"
                        and s.get_is_completed()
                    ]

                    touched_cities = set()
                    for card, _ in structure.card_sides:
                        neighbors = card.get_neighbors().values()
                        touched_cards = set([card])
                        touched_cards.update([n for n in neighbors if n])

                        for city_structure in completed_cities:
                            for city_card, _ in city_structure.card_sides:
                                if city_card in touched_cards:
                                    touched_cities.add(city_structure)
                                    break

                    field_score = len(touched_cities) * 6
                    score += field_score * self._preset["field_multiplier"]

                    field_size = len(structure.card_sides)
                    if field_size > 8:
                        score += 35.0
                    elif field_size > 6:
                        score += 25.0
                    elif field_size > 4:
                        score += 15.0
                    elif field_size > 2:
                        score += 10.0

        return score

    def _should_conserve_figure(self, game_session: 'GameSession') -> bool:
        """
        Determine if the AI should conserve meeples based on game state.
        
        Args:
            game_session: The current game session
            
        Returns:
            True if the AI should conserve meeples, False otherwise
        """
        return len(self.figures) <= self._preset["conservation_threshold"]

    def _handle_figure_placement_advanced(self, game_session: 'GameSession',
                                          target_x: int,
                                          target_y: int) -> None:
        """
        Handle meeple placement using advanced strategic evaluation with preset configuration.
        
        Args:
            game_session: The current game session
            target_x: X coordinate where the card was placed
            target_y: Y coordinate where the card was placed
        """
        card = game_session.last_placed_card
        terrains = card.get_terrains()
        best_direction = None
        best_score = float('-inf')

        should_conserve = self._should_conserve_figure(game_session)

        for direction in terrains.keys():
            if direction == "C" and terrains.get(direction) != "monastery":
                continue

            structure = game_session.structure_map.get(
                (target_x, target_y, direction))
            if structure and not structure.get_figures() and len(
                    self.figures) > 0:
                score = self._evaluate_figure_placement_advanced(
                    game_session, target_x, target_y, direction)

                completion_ratio = self._calculate_structure_completion_ratio(structure)

                if completion_ratio > 0.8:
                    score += 120.0
                elif completion_ratio > 0.6:
                    score += 80.0
                elif completion_ratio > 0.4:
                    score += 50.0
                elif completion_ratio > 0.2:
                    score += 25.0

                if should_conserve:
                    if not structure.get_is_completed(
                    ) and completion_ratio < 0.5:
                        score *= 0.5

                    if structure.get_structure_type() == "Field":
                        score *= 0.7

                if score > best_score:
                    best_score = score
                    best_direction = direction

        if best_direction and best_score > 0:
            threshold = self._preset[
                "placement_threshold"] if should_conserve else 0.0

            if best_score >= threshold:
                if game_session.play_figure(self, target_x, target_y,
                                            best_direction):
                    logger.debug(
                        f"Player {self.name} placed meeple on {best_direction} (score: {best_score}, conserving: {should_conserve})"
                    )
                    # Check for completed structures and score them immediately
                    self._check_and_score_completed_structures(game_session)
                    game_session.next_turn()
                    return
            else:
                logger.debug(
                    f"Player {self.name} chose not to place meeple (score: {best_score} < threshold: {threshold}, conserving: {should_conserve})"
                )

        logger.info(
            f"Player {self.name} couldn't place meeple anywhere or chose not to"
        )
        game_session.skip_current_action()

    def _evaluate_figure_placement_advanced(self, game_session: 'GameSession',
                                            x: int, y: int,
                                            direction: str) -> float:
        """
        Evaluate the score of a meeple placement using preset configuration.
        
        Args:
            game_session: The current game session
            x: X coordinate for placement
            y: Y coordinate for placement
            direction: The direction to place the meeple
            
        Returns:
            A score representing the desirability of this meeple placement
        """

        def evaluate_figure_placement():
            score = 0.0
            structure = self._get_candidate_structure(game_session, x, y, direction)

            if not structure:
                return 0.0

            completion_ratio = self._calculate_structure_completion_ratio(structure)
            score += completion_ratio * self._preset["figure_opportunity"]

            structure_type = structure.get_structure_type()
            if structure_type == "City":
                score += self._evaluate_city_figure_placement(structure)
            elif structure_type == "Road":
                score += self._evaluate_road_figure_placement(structure)
            elif structure_type == "Monastery":
                score += self._evaluate_monastery_figure_placement(structure)
            elif structure_type == "Field":
                score += self._evaluate_field_figure_placement(
                    game_session, structure)

            return score

        return self._evaluate_figure_cached(x, y, direction, "advanced",
                                            evaluate_figure_placement)

    def _evaluate_city_figure_placement(self, structure: 'Structure') -> float:
        """Evaluate city figure placement scoring."""
        score = 0.0

        total_sides = len(structure.card_sides)

        if total_sides <= 4:
            score += 60.0
        elif total_sides <= 6:
            score += 40.0
        elif total_sides <= 8:
            score += 25.0

        return score

    def _evaluate_road_figure_placement(self, structure: 'Structure') -> float:
        """Evaluate road figure placement scoring."""
        score = 0.0

        total_sides = len(structure.card_sides)

        if total_sides <= 3:
            score += 50.0
        elif total_sides <= 5:
            score += 35.0
        elif total_sides <= 7:
            score += 20.0

        return score

    def _evaluate_monastery_figure_placement(self,
                                             structure: 'Structure') -> float:
        """Evaluate monastery meeple placement scoring."""
        score = 0.0

        completion_ratio = self._calculate_structure_completion_ratio(
            structure)

        if completion_ratio > 0.6:
            score += 60.0
        elif completion_ratio > 0.4:
            score += 40.0
        elif completion_ratio > 0.2:
            score += 25.0

        return score

    def _evaluate_field_figure_placement(self, game_session: 'GameSession',
                                         structure: 'Structure') -> float:
        """Evaluate field meeple placement scoring."""
        score = 0.0

        completed_cities = [
            s for s in game_session.structures
            if s.get_structure_type() == "City" and s.get_is_completed()
        ]

        touched_cities = set()
        for card, _ in structure.card_sides:
            neighbors = card.get_neighbors().values()
            touched_cards = set([card])
            touched_cards.update([n for n in neighbors if n])

            for city_structure in completed_cities:
                for city_card, _ in city_structure.card_sides:
                    if city_card in touched_cards:
                        touched_cities.add(city_structure)
                        break

        score += len(touched_cities) * 8.0

        if len(structure.card_sides) > 8:
            score += 50.0
        elif len(structure.card_sides) > 6:
            score += 35.0
        elif len(structure.card_sides) > 4:
            score += 25.0

        return score

    def _handle_figure_placement_simple(self, game_session: 'GameSession',
                                        target_x: int, target_y: int) -> None:
        """
        Handle meeple placement with simple logic.
        
        Args:
            game_session: The current game session
            target_x: X coordinate where the card was placed
            target_y: Y coordinate where the card was placed
        """
        card = game_session.last_placed_card
        terrains = card.get_terrains()
        best_direction = None
        best_score = float('-inf')

        for direction in terrains.keys():
            if direction == "C" and terrains.get(direction) != "monastery":
                continue

            structure = game_session.structure_map.get(
                (target_x, target_y, direction))
            if structure and not structure.get_figures():
                score = self._evaluate_figure_placement(
                    game_session, target_x, target_y, direction)
                if score > best_score:
                    best_score = score
                    best_direction = direction

        if best_direction and best_score > 0:
            if game_session.play_figure(self, target_x, target_y,
                                        best_direction):
                logger.debug(
                    f"Player {self.name} placed meeple on {best_direction} (score: {best_score})"
                )
                self._check_and_score_completed_structures(game_session)
                game_session.next_turn()
                return

        logger.info(
            f"Player {self.name} couldn't place meeple anywhere or chose not to"
        )
        game_session.skip_current_action()

    def _check_and_score_completed_structures(
            self, game_session: 'GameSession') -> None:
        """Check for completed structures and score them immediately."""
        logger.debug("Checking completed structures...")
        for structure in game_session.structures:
            structure.check_completion()
            if structure.get_is_completed():
                logger.debug(
                    f"Structure {structure.structure_type} is completed!")
                game_session.score_structure(structure)

    def _evaluate_figure_placement(self, game_session: 'GameSession', x: int,
                                   y: int, direction: str) -> float:
        """
        Evaluate the score of a meeple placement.
        
        Args:
            game_session: The current game session
            x: X coordinate for placement
            y: Y coordinate for placement
            direction: The direction to place the meeple
            
        Returns:
            A score representing the desirability of this meeple placement
        """
        score = 0.0
        structure = self._get_candidate_structure(game_session, x, y, direction)

        if not structure:
            return 0.0

        if structure.get_is_completed():
            score += 100.0
        else:
            completion_ratio = self._calculate_structure_completion_ratio(structure)
            score += completion_ratio * 50.0

        return score

    def serialize(self) -> dict:
        """
        Serialize the AI player to a dictionary.
        
        Returns:
            Dictionary containing serialized AI player data
        """
        base_data = super().serialize()
        base_data["difficulty"] = self._difficulty
        return base_data

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> Optional['AIPlayer']:
        """
        Deserialize an AI player from a dictionary.
        
        Args:
            data: Dictionary containing player data
            
        Returns:
            An AIPlayer instance if successful, None otherwise
        """
        try:
            name = str(data["name"])
            index = int(data["index"])
            color = str(data["color"])
            is_ai = bool(data.get("is_ai", True))
            score = int(data.get("score", 0))
            figures_remaining = int(data.get("figures_remaining", 7))
            is_human = bool(data.get("is_human", False))
            difficulty = str(data.get("difficulty", "NORMAL"))

            player = AIPlayer(name=name,
                              index=index,
                              color=color,
                              difficulty=difficulty)
            player.score = score
            player.figures = [Figure(player) for _ in range(figures_remaining)]
            logger.debug(
                f"Deserialized AI player {name} with {len(player.figures)} figures"
            )
            return player
        except (KeyError, ValueError, TypeError) as e:
            logger.error(
                f"Failed to deserialize AIPlayer object: {e}\nData: {data}")
            return None
