# Preprocessed Data

Tài liệu này mô tả các biến được lưu sau bước tiền xử lý dữ liệu

## Data Description

| Variable                | Description                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `train_edges`           | Positive user–item interactions dùng để huấn luyện mô hình. Mỗi dòng có dạng `(user_id, item_id)`.                 |
| `val_edges`             | Validation interaction của mỗi user, dùng để đánh giá trong quá trình phát triển mô hình.                          |
| `test_edges`            | Test interaction của mỗi user, dùng để đánh giá cuối cùng.                                                         |
| `edge_index_train`      | Bidirectional graph edge index được xây dựng từ `train_edges`, sử dụng cho quá trình message passing của LightGCN. |
| `item_degree`           | Số lượng training interactions của từng item (degree của item trong training graph).                               |
| `item_popularity_group` | Nhóm popularity của từng item, được xác định dựa trên `item_degree`.                                               |

## Data Schema

| Variable                | Python Type    | Data Type     | Shape          |
| ----------------------- | -------------- | ------------- | -------------- |
| `train_edges`           | `torch.Tensor` | `torch.int64` | `(563204, 2)`  |
| `val_edges`             | `torch.Tensor` | `torch.int64` | `(6034, 2)`    |
| `test_edges`            | `torch.Tensor` | `torch.int64` | `(6034, 2)`    |
| `edge_index_train`      | `torch.Tensor` | `torch.int64` | `(2, 1126408)` |
| `item_degree`           | `torch.Tensor` | `torch.int64` | `(3533,)`      |
| `item_popularity_group` | `torch.Tensor` | `torch.int64` | `(3533,)`      |

## Notes

* `train_edges`, `val_edges`, và `test_edges` đều lưu các cặp `(user_id, item_id)` dưới dạng tensor có kích thước `(num_edges, 2)`.
* `edge_index_train` có định dạng `(2, num_edges_bidirectional)` theo chuẩn của **PyTorch Geometric** để phục vụ graph message passing.
* `item_degree[i]` là số lần item `i` xuất hiện trong tập huấn luyện.
* `item_popularity_group[i]` là nhóm popularity tương ứng của item `i