from src.chess_tokenizers import UciTileTokenizer

tok = UciTileTokenizer(upper_promotions=True)

assert tok("e2e4Q b7b8N e2e7 a1", add_special_tokens=True) == {
    "input_ids": [1, 16, 32, 68, 53, 61, 71, 16, 56, 4],
    "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}

assert tok("e2e4Q b7b8N e2e7 a1")["input_ids"] == [1, 16, 32, 68, 53, 61, 71, 16, 56, 4]

try:
    tok.compute_ply_end_indices(tok("e2"))
except AssertionError:
    pass
else:
    raise AssertionError("Expected AssertionError for missing offsets")

assert tok.compute_ply_end_indices(tok("h8", return_offsets_mapping=True)) == [1]
assert tok.compute_ply_end_indices(tok("e2", return_offsets_mapping=True)) == [1]
assert tok.compute_ply_end_indices(tok("", return_offsets_mapping=True)) == []

assert tok.compute_ply_end_indices(tok("b1c3", return_offsets_mapping=True)) == [2]
assert tok.compute_ply_end_indices(tok("b1c3Q", return_offsets_mapping=True)) == [3]
assert tok.compute_ply_end_indices(tok("QBNR", return_offsets_mapping=True)) == [4]
assert tok.compute_ply_end_indices(tok("QQ", return_offsets_mapping=True)) == [2]
assert tok.compute_ply_end_indices(tok("b1b2 Q", return_offsets_mapping=True)) == [2, 3]
assert tok.compute_ply_end_indices(tok("b1b2 QQ", return_offsets_mapping=True)) == [2, 4]
assert tok.compute_ply_end_indices(tok("b1b2 QQQ", return_offsets_mapping=True)) == [2, 5]

# the following shows the inconsistency in the final token handling
assert tok.compute_ply_end_indices(tok("b1 b2 b3 b4 b5", return_offsets_mapping=True)) == [
    1,
    2,
    3,
    4,
    5,
]
assert tok.compute_ply_end_indices(tok("b1 b2 b3 b4 b5b8", return_offsets_mapping=True)) == [
    1,
    2,
    3,
    4,
    6,
]

assert tok.compute_ply_end_indices(tok([""], return_offsets_mapping=True)) == [[]]
assert tok.compute_ply_end_indices(tok(["e2"], return_offsets_mapping=True)) == [[1]]
assert tok.compute_ply_end_indices(tok(["e2", "Q"], return_offsets_mapping=True)) == [
    [1],
    [1],
]

assert tok.compute_ply_end_indices(
    tok(
        "e2e4 d7d5 e4d5 e7e6 d5e6 d8g5 e6e7 g5f6 e7f8Q",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
) == [2, 4, 6, 8, 10, 12, 14, 16, 19]

encoding = tok(
    "e2e4Q b7b8N e2e7 a1", return_offsets_mapping=True
)  # special tokens are added by default
indices = tok.compute_ply_end_indices(encoding)
assert tok.compute_players(encoding, according_to="output") == [
    True,
    True,
    True,
    False,
    False,
    False,
    True,
    True,
    False,
    False,
]
assert tok.compute_players(encoding, according_to="input") == [
    True,
    True,
    True,
    True,
    False,
    False,
    False,
    True,
    True,
    False,
]
assert tok.compute_players_from_indices(indices) == [
    True,
    True,
    True,
    True,
    False,
    False,
    False,
    True,
    True,
    False,
]
encoding = tok("e2e4Q b7b8N e2e7 a1", add_special_tokens=False, return_offsets_mapping=True)
assert tok.compute_players(encoding, according_to="output") == [
    True,
    True,
    False,
    False,
    False,
    True,
    True,
    False,
    False,
]
assert tok.compute_players(encoding, according_to="input") == [
    True,
    True,
    True,
    False,
    False,
    False,
    True,
    True,
    False,
]
