import pygame
import webbrowser
import typing
from ui.scene import Scene
from ui.components.button import Button
from ui.utils.draw import draw_rect_alpha
from game_state import GameState
from ui import theme


class HelpScene(Scene):
    """Scene displaying help and controls for the game."""

    def __init__(self, screen: pygame.Surface,
                 switch_scene_callback: typing.Callable) -> None:
        super().__init__(screen, switch_scene_callback)
        self.font = theme.get_font("title", theme.THEME_FONT_SIZE_SCENE_TITLE)
        self.section_header_font = theme.get_font(
            "section_header", theme.THEME_FONT_SIZE_SECTION_HEADER
        )
        self.button_font = theme.get_font(
            "button", theme.THEME_FONT_SIZE_BUTTON
        )
        self.text_font = theme.get_font("body", theme.THEME_FONT_SIZE_BODY)
        self.controls_font = self.text_font
        self.scroll_offset = 0
        self.max_scroll = 0
        self.scroll_speed = 30
        self.header_height = 0
        self._title_text = "How to Play"
        self._title_surface: pygame.Surface | None = None
        self._cached_title_text: str | None = None
        self._section_header_surfaces: dict[str, pygame.Surface] = {}
        self._section_titles_cache: tuple[str, ...] = ()
        self._section_lines_cache: tuple[str, ...] = ()
        self._line_surfaces: dict[tuple[int, str, tuple[int, int, int]],
                                  pygame.Surface] = {}
        self.sections = [
            ("Objective of the Game", [
                "Your goal is to score more points than the other players.",
                "You earn points by placing cards, completing structures,",
                "and placing your figures at the right time.",
                "The game ends when there are no cards left in the deck.",
            ]),
            ("Starting a New Game", [
                "1) From the main menu, select New Game.",
                "2) Set player names and choose which players are controlled by AI.",
                "3) Select the AI difficulty and available card sets.",
                "4) Press Start Game to begin.",
                "5) The starting card is placed automatically in the center of the board.",
            ]),
            ("Controls", [
                "- LMB: place cards and figures / rotate the card by clicking its sidebar preview.",
                "- RMB: rotate the current card.",
                "- WASD / Arrow Keys: move around the board.",
                "- SPACE: skip figure placement or discard a card.",
                "- TAB: show or hide the game log.",
                "- ESC: return to the main menu.",
            ]),
            ("Turn Structure", [
                "Each turn is divided into two phases.",
                "",
                "Phase 1: Place a Card",
                "- Draw one card and place it so all touching edges match.",
                "- You can rotate the card before placing it.",
                "- If several positions are valid, you may choose any of them.",
                "- If no valid placement exists, press SPACE to discard the card.",
                "",
                "Phase 2: Place a Figure (Optional)",
                "- The figure must be placed on the card you just played.",
                "- The chosen region must not already contain another figure.",
                "- Only one figure may be placed per turn.",
                "- You may skip this phase by pressing SPACE.",
                "- Figures remain on the board until the structure is completed.",
            ]),
            ("Scoring", [
                "Points are awarded automatically when a structure is completed.",
                "When a structure is finished, all figures on it are returned to their owners.",
                "Unfinished structures are scored at the end of the game.",
                "Current scores are always visible in the sidebar.",
            ]),
            ("End of the Game", [
                "The game ends when the card deck is empty.",
                "All unfinished structures are then scored.",
                "Final scores and player rankings are displayed.",
            ]),
            ("Hosting a Network Game", [
                "1) Open New Game and set Network Mode to Host.",
                "2) Configure players and game settings.",
                "3) Your local IP address is shown automatically.",
                "4) Share the IP address and port with other players.",
                "5) Wait until all players have connected.",
                "6) Press Start Game to begin the match.",
            ]),
            ("Joining a Network Game", [
                "1) Open New Game and set Network Mode to Client.",
                "2) Enter the host’s IP address and port.",
                "3) Press Start Game to connect.",
                "4) Wait in the lobby until the host starts the game.",
            ]),
        ]

        self.section_headers_layout: list[tuple[str, int]] = []
        self.section_body_layout: list[tuple[str, int]] = []
        self.section_divider_layout: list[int] = []

        self.rules_button = Button(pygame.Rect(0, 0, 0, 60), "Wiki",
                                   self.button_font)
        self.back_button = Button(pygame.Rect(0, 0, 0, 60), "Back",
                                  self.button_font)

        self._layout_controls()

    def _get_line_style(self, line: str) -> tuple[pygame.font.Font, tuple]:
        return self.controls_font, theme.THEME_TEXT_COLOR_LIGHT

    def _get_line_surface(
        self,
        font: pygame.font.Font,
        line: str,
        color: tuple[int, int, int],
    ) -> pygame.Surface:
        cache_key = (id(font), line, color)
        cached = self._line_surfaces.get(cache_key)
        if cached is None:
            cached = font.render(line, True, color)
            self._line_surfaces[cache_key] = cached
        return cached

    def _set_text_rect(self, line: str, font: pygame.font.Font, y: int,
                       padding: int) -> tuple[int, int]:
        _, line_height = font.size(line)
        return y, y + line_height + padding

    def _set_component_center(self, component, center_x: int, y: int,
                              padding: int) -> int:
        width, height = component.rect.size
        component.rect = pygame.Rect(0, 0, width, height)
        component.rect.center = (center_x, y + height // 2)
        return y + height + padding

    def _get_content_bounds(self) -> tuple[int, int]:
        screen_width = self.screen.get_width()
        max_width = min(
            int(screen_width * 0.7),
            theme.THEME_HELP_MAX_WIDTH,
        )
        content_left = (screen_width - max_width) // 2
        content_right = content_left + max_width
        return content_left, content_right

    def _layout_controls(self) -> None:
        section_gap = theme.THEME_LAYOUT_SECTION_GAP
        line_gap = theme.THEME_LAYOUT_LINE_GAP
        divider_padding = theme.THEME_SECTION_DIVIDER_PADDING
        button_center_x = self.screen.get_width() // 2
        self.header_height = self._get_scene_header_height(
            self.font.get_height()
        )
        current_y = self.header_height + section_gap

        self.section_headers_layout.clear()
        self.section_body_layout.clear()
        self.section_divider_layout.clear()
        for index, (section_title, section_lines) in enumerate(self.sections):
            if index > 0:
                current_y += section_gap
            self.section_headers_layout.append((section_title, current_y))
            current_y += self.section_header_font.get_height() + line_gap
            for line in section_lines:
                line_y, current_y = self._set_text_rect(
                    line, self.controls_font, current_y, line_gap)
                self.section_body_layout.append((line, line_y))
            current_y += divider_padding
            divider_y = current_y
            self.section_divider_layout.append(divider_y)
            current_y += divider_padding

        current_y += theme.THEME_LAYOUT_BUTTON_SECTION_GAP
        current_y = self._set_component_center(
            self.rules_button,
            button_center_x,
            current_y,
            theme.THEME_LAYOUT_VERTICAL_GAP,
        )
        self._set_component_center(
            self.back_button,
            button_center_x,
            current_y,
            theme.THEME_LAYOUT_VERTICAL_GAP,
        )

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        """Handle events for the help scene."""
        self._apply_scroll(events)
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.switch_scene(GameState.MENU)
            if event.type in (pygame.MOUSEMOTION,
                              pygame.MOUSEBUTTONDOWN,
                              pygame.MOUSEBUTTONUP):
                self.back_button.handle_event(event,
                                              y_offset=self.scroll_offset)
                self.rules_button.handle_event(event,
                                               y_offset=self.scroll_offset)
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.back_button.handle_event(event,
                                              y_offset=self.scroll_offset)
                self.rules_button.handle_event(event,
                                               y_offset=self.scroll_offset)
                if self.back_button._is_clicked(event.pos,
                                                y_offset=self.scroll_offset):
                    self.switch_scene(GameState.MENU)
                elif self.rules_button._is_clicked(
                        event.pos, y_offset=self.scroll_offset):
                    webbrowser.open("https://wikicarpedia.com/car/Base_game")

    def draw(self) -> None:
        """Draw the help scene."""
        self._draw_background(
            background_color=theme.THEME_HELP_BACKGROUND_COLOR,
            image_name=theme.THEME_HELP_BACKGROUND_IMAGE,
            scale_mode=theme.THEME_HELP_BACKGROUND_SCALE_MODE,
            tint_color=theme.THEME_HELP_BACKGROUND_TINT_COLOR,
            blur_radius=theme.THEME_HELP_BACKGROUND_BLUR_RADIUS,
        )
        if (self._title_surface is None
                or self._cached_title_text != self._title_text):
            self._title_surface = self.font.render(
                self._title_text, True, theme.THEME_SCENE_HEADER_TEXT_COLOR
            )
            self._cached_title_text = self._title_text
        offset_y = self.scroll_offset
        content_left, content_right = self._get_content_bounds()
        section_titles = tuple(title for title, _ in self.sections)
        section_lines = tuple(
            line for _, lines in self.sections for line in lines
        )
        if section_titles != self._section_titles_cache:
            self._section_titles_cache = section_titles
            self._section_header_surfaces.clear()
        if section_lines != self._section_lines_cache:
            self._section_lines_cache = section_lines
            self._line_surfaces.clear()

        for section_title, header_y in self.section_headers_layout:
            section_label = self._section_header_surfaces.get(section_title)
            if section_label is None:
                section_label = self.section_header_font.render(
                    section_title, True, theme.THEME_SECTION_HEADER_COLOR)
                self._section_header_surfaces[section_title] = section_label
            section_label_rect = section_label.get_rect()
            section_label_rect.center = (
                self.screen.get_width() // 2,
                header_y + offset_y + section_label_rect.height // 2,
            )
            self.screen.blit(section_label, section_label_rect)

        for line, line_y in self.section_body_layout:
            font, color = self._get_line_style(line)
            text_surface = self._get_line_surface(font, line, color)
            draw_rect = text_surface.get_rect()
            draw_rect.left = content_left
            draw_rect.y = line_y + offset_y
            if draw_rect.bottom > 0 and draw_rect.top < self.screen.get_height(
            ):
                self.screen.blit(text_surface, draw_rect)

        divider_height = 2
        divider_width = content_right - content_left
        for divider_y in self.section_divider_layout:
            draw_y = divider_y + offset_y
            if (draw_y + divider_height > 0
                    and draw_y < self.screen.get_height()):
                divider_rect = pygame.Rect(
                    content_left,
                    draw_y,
                    divider_width,
                    divider_height,
                )
                draw_rect_alpha(
                    self.screen,
                    theme.THEME_SECTION_DIVIDER_COLOR,
                    divider_rect,
                )

        self.rules_button.draw(self.screen, y_offset=offset_y)
        self.back_button.draw(self.screen, y_offset=offset_y)
        self._draw_scene_header(self._title_surface)
        self.max_scroll = max(
            self.screen.get_height(),
            self.back_button.rect.bottom + theme.THEME_LAYOUT_SECTION_GAP * 2,
        )

    def refresh_theme(self, theme_name: str | None = None) -> None:
        """Refresh fonts and component styling after theme changes."""
        super().refresh_theme(theme_name)
        self.font = theme.get_font("title", theme.THEME_FONT_SIZE_SCENE_TITLE)
        self.section_header_font = theme.get_font(
            "section_header", theme.THEME_FONT_SIZE_SECTION_HEADER
        )
        self.button_font = theme.get_font(
            "button", theme.THEME_FONT_SIZE_BUTTON
        )
        self.text_font = theme.get_font("body", theme.THEME_FONT_SIZE_BODY)
        self.controls_font = theme.get_font("body", theme.THEME_FONT_SIZE_BODY)
        self._title_surface = None
        self._cached_title_text = None
        self._section_header_surfaces.clear()
        self._line_surfaces.clear()
        self.rules_button.set_font(self.button_font)
        self.rules_button.apply_theme()
        self.back_button.set_font(self.button_font)
        self.back_button.apply_theme()
        self._layout_controls()
