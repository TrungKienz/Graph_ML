# Explain preprocess data

- `train_edges`: các positive user–movie interactions dùng cho training.
- `val_edges`: interaction validation của từng user.
- `test_edges`: interaction test của từng user.
- `edge_index_train`: bidirectional graph edge index dùng cho LightGCN message passing.
- `item_degree`: số lượng training interactions của từng item.
- `item_popularity_group`: nhóm popularity của từng item.