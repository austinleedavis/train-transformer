import math
import warnings
from io import StringIO
from typing import Any, Callable, Iterable, List, Union

import chess
import chess.pgn
import chess.svg
import regex as re

UCI_TOKEN_PATTERN = re.compile(r"[a-h]\d|[QBRN]")


def get_board_position_change_indices(
    token_offsets: list[tuple[int, int]], n_pos: int
) -> list[int]:
    """Finds the token indices where a tokenized UCI transcript resolves into different board
    positions.

    This function determines the points at which the board state changes between moves in a
    sequence of UCI (Universal Chess Interface) moves. It uses the offsets from a UciTileTokenizer
    to identify these points.

    Args:
        offsets (list[tuple[int, int]]): A list of tuples representing the start and end offsets
                                         of tokens in the UCI transcript.
        n_pos (int): The number of positions (moves) in the UCI transcript.

    Returns:
        list[int]: A list of indices indicating where the board state changes between moves.
    """
    split_indicies = [
        p for p in range(n_pos - 1) if token_offsets[p, 1] != token_offsets[p + 1, 0]
    ] + [int(n_pos) - 1]
    return split_indicies


def truncate_illegal_moves(uci_moves: Union[str, Iterable[str]]) -> tuple[str, int]:
    if isinstance(uci_moves, str):
        uci_moves = uci_moves.split(" ")

    board = chess.Board()
    for i, move in enumerate(uci_moves):
        try:
            move_obj = board.parse_uci(move.lower())
        except Exception as e:
            return " ".join(uci_moves[:i]), i, move
        board.push(move_obj)

    return " ".join(uci_moves[: i + 1]), i + 1, ""


def uci_to_board(
    uci_moves: Union[str, Iterable],
    *,
    force=False,
    fail_silent=False,
    verbose=True,
    as_board_stack=False,
    map_function: Callable = lambda x: x,
    reset_halfmove_clock=False,
) -> Union[chess.Board, List[Union[chess.Board, Any]]]:
    """Returns a chess.Board object from a string of UCI moves
    Params:
        force: If true, illegal moves are forcefully made. O/w, the error is thrown
        verbose: Alert user via prints that illegal moves were attempted."""
    board = chess.Board()
    forced_moves = []
    did_force = False
    board_stack = [map_function(board.copy())]

    if isinstance(uci_moves, str):
        uci_moves = uci_moves.lower().split(" ")

    for i, move in enumerate(uci_moves):
        try:
            move_obj = board.parse_uci(move)
            if reset_halfmove_clock:
                board.halfmove_clock = 0
            board.push(move_obj)
        except (chess.IllegalMoveError, chess.InvalidMoveError) as ex:
            if force:
                did_force = True
                forced_moves.append((i, move))
                piece = board.piece_at(chess.parse_square(move[:2]))
                board.set_piece_at(chess.parse_square(move[:2]), None)
                board.set_piece_at(chess.parse_square(move[2:4]), piece)
            elif fail_silent:
                if as_board_stack:
                    return board_stack
                else:
                    return map_function(board)
            else:
                if verbose:
                    print(f"Failed on (move_id, uci): ({i},{move})")
                    if as_board_stack:
                        return board_stack
                    else:
                        return map_function(board)
                else:
                    raise ex
        board_stack.append(map_function(board.copy()))
    if verbose and did_force:
        print(f"Forced (move_id, uci): {forced_moves}")

    if as_board_stack:
        return board_stack
    else:
        return map_function(board)


def pgn_to_uci(pgn_string: str):
    """Converts a pgn string into uci notation. Example usage: ```

    >>> pgn_to_uci('1.e4 e5 2.Nf3 Nc6 3.Bb5')
    'e2e4 e7e5 g1f3 b8c6 f1b5'
    ```
    """

    pgn_io = StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)
    return " ".join([m.uci() for m in game.mainline_moves()])


def pgn_game_to_uci_moves(pgn_game: chess.pgn.Game):
    return " ".join([m.uci() for m in pgn_game.mainline_moves()])


def uci_to_pgn(
    uci_string: str,
    headers=dict(
        Event="?",
        Site="?",
        Date="????.??.??",
        Round="?",
        White="?",
        Black="?",
        Result="*",
    ),
):
    """Converts a uci string into pgn.

    Example usage (**using print**):
    ```
    >>> print(uci_to_pgn('e2e4 e7e5 g1f3 b8c6 f1b5'))
    [Event "?"]
    [Site "?"]
    [Date "????.??.??"]
    [Round "?"]
    [White "?"]
    [Black "?"]
    [Result "*"]

    1. e4 e5 2. Nf3 Nc6 3. Bb5 *
    ```
    """

    game_pgn = chess.pgn.Game()
    game_pgn.headers.update(**headers)
    node = game_pgn
    for ply, move in enumerate(uci_string.lower().split()):
        try:
            node = node.add_variation(chess.Move.from_uci(move))
        except AssertionError as ex:
            warnings.warn(f"Warning! UCI cannot be converted to PGN on ply {ply}: '{move}'.")
            break
    return game_pgn


def win2cp(win_percent):
    """Convert a win percentage to centipawn evaluation.

    Parameters:
    win_percent (float): The win percentage, a value between 0 and 1.

    Returns:
    float: The centipawn evaluation. Returns positive infinity for a win percentage of 1,
           negative infinity for a win percentage of 0, and a calculated centipawn value otherwise.

    (Formula derived from https://lichess.org/page/accuracy)
    """
    if win_percent == 1:
        return math.inf
    if win_percent == 0:
        return -math.inf

    return (6250000 * math.log(-win_percent / (win_percent - 1))) / 23013


def cp2win(centipawns):
    """Convert centipawn evaluation to win percentage.

    Parameters:
    centipawns (float): The evaluation in centipawns.

    Returns:
    float: The win percentage corresponding to the centipawn evaluation.

    (Formula derived from https://lichess.org/page/accuracy)
    """

    return 1 - 1 / (math.exp(0.00368208 * centipawns) + 1)


def accuracy(win_percent_before, win_percent_after):
    """Computes the accuracy of a move as a function of the win percentage before and after the
    move.

    Parameters:
    win_percent_before (float): The win percentage before the move.
    win_percent_after (float): The win percentage after the move.

    Returns:
    float: The computed accuracy of the move.
    """

    delta = win_percent_before - win_percent_after
    return 103.1668 * math.exp(-0.04354 * delta) - 3.1669
