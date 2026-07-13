import random

from eyetracker.config import TrackerConfig
from eyetracker.game import STATE_OVER, STATE_PLAYING, STATE_READY, EventType, GazePopGame


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_game(**overrides):
    cfg = TrackerConfig(**overrides)
    clock = FakeClock()
    game = GazePopGame(cfg, width=1000, height=1000, rng=random.Random(1), clock=clock)
    return game, clock


def _tick(game, clock, gaze, seconds, steps=5):
    """Advance `seconds` of game time in small steps (dt is clamped per update)."""
    events = []
    for _ in range(steps):
        clock.advance(seconds / steps)
        events += game.update(gaze)
    return events


def test_starts_in_ready_with_a_center_button():
    game, _ = _make_game()
    assert game.state == STATE_READY
    assert len(game.targets) == 1
    assert game.targets[0].is_button


def test_dwelling_the_start_button_begins_the_game():
    game, clock = _make_game(game_dwell_sec=0.5)
    events = _tick(game, clock, gaze=(0.5, 0.5), seconds=0.6)
    assert game.state == STATE_PLAYING
    assert any(e.type is EventType.START for e in events)


def test_looking_away_does_not_start_the_game():
    game, clock = _make_game(game_dwell_sec=0.5)
    _tick(game, clock, gaze=(0.05, 0.05), seconds=1.0)  # nowhere near the center button
    assert game.state == STATE_READY


def test_holding_gaze_on_an_orb_pops_it_and_scores():
    game, clock = _make_game(game_dwell_sec=0.5)
    _tick(game, clock, gaze=(0.5, 0.5), seconds=0.6)  # start
    assert game.state == STATE_PLAYING
    orb = game.targets[0]
    events = _tick(game, clock, gaze=(orb.x, orb.y), seconds=0.6)
    assert any(e.type is EventType.POP for e in events)
    assert game.score > 0
    assert game.popped == 1


def test_consecutive_pops_build_a_combo_multiplier():
    game, clock = _make_game(game_dwell_sec=0.3, game_combo_window_sec=5.0)
    _tick(game, clock, gaze=(0.5, 0.5), seconds=0.4)  # start
    scores = []
    for _ in range(3):
        orb = game.targets[0]
        _tick(game, clock, gaze=(orb.x, orb.y), seconds=0.4)
        scores.append(game.score)
    # Each pop is worth more than the last because the combo multiplier grew.
    gains = [scores[0], scores[1] - scores[0], scores[2] - scores[1]]
    assert gains[1] > gains[0]
    assert gains[2] > gains[1]
    assert game.best_combo >= 3


def test_combo_resets_after_the_window_lapses():
    game, clock = _make_game(game_dwell_sec=0.3, game_combo_window_sec=1.0)
    _tick(game, clock, gaze=(0.5, 0.5), seconds=0.4)  # start
    orb = game.targets[0]
    _tick(game, clock, gaze=(orb.x, orb.y), seconds=0.4)
    assert game.combo == 1
    _tick(game, clock, gaze=None, seconds=1.5)  # idle past the combo window
    assert game.combo == 0


def test_round_ends_after_the_duration():
    game, clock = _make_game(game_dwell_sec=0.3, game_duration_sec=2.0)
    _tick(game, clock, gaze=(0.5, 0.5), seconds=0.4)  # start
    events = _tick(game, clock, gaze=None, seconds=2.5, steps=30)
    assert game.state == STATE_OVER
    assert any(e.type is EventType.GAME_OVER for e in events)
    assert game.time_left == 0.0


def test_ignored_orb_expires_and_breaks_combo():
    game, clock = _make_game(game_dwell_sec=0.3, game_target_lifetime_sec=1.0, game_duration_sec=30.0)
    _tick(game, clock, gaze=(0.5, 0.5), seconds=0.4)  # start
    orb = game.targets[0]
    _tick(game, clock, gaze=(orb.x, orb.y), seconds=0.4)  # pop -> combo 1
    assert game.combo == 1
    _tick(game, clock, gaze=None, seconds=1.3, steps=20)  # let a fresh orb expire, idle
    assert game.missed >= 1
    assert game.combo == 0


def test_never_exceeds_max_targets():
    game, clock = _make_game(game_dwell_sec=1.0, game_max_targets=3, game_spawn_interval_sec=0.1)
    _tick(game, clock, gaze=(0.5, 0.5), seconds=1.1)  # start (dwell 1.0)
    _tick(game, clock, gaze=None, seconds=3.0, steps=30)  # let spawns accumulate
    assert len(game.targets) <= 3


def test_game_over_button_restarts_the_round():
    game, clock = _make_game(game_dwell_sec=0.3, game_duration_sec=1.0)
    _tick(game, clock, gaze=(0.5, 0.5), seconds=0.4)  # start
    _tick(game, clock, gaze=None, seconds=1.2, steps=20)  # let time run out
    assert game.state == STATE_OVER
    _tick(game, clock, gaze=(0.5, 0.5), seconds=0.4)  # dwell the replay button
    assert game.state == STATE_PLAYING
    assert game.score == 0  # fresh round
