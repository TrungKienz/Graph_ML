# RAW DATA AFTER PREPROCESS AND SPLIT

Thư mục này chứa dữ liệu sau khi tiền xử lý và chia thành các tập **train**, **validation**, và **test**. Dữ liệu vẫn được lưu dưới dạng **Pandas DataFrame** (`.parquet`)

## Files

| File            | Description                                                                            |
| --------------- | -------------------------------------------------------------------------------------- |
| `train.parquet` | Training interactions.                                                                 |
| `val.parquet`   | Validation interactions. Mỗi user chỉ có một interaction được giữ lại để đánh giá.     |
| `test.parquet`  | Test interactions. Mỗi user chỉ có một interaction được giữ lại để đánh giá cuối cùng. |

## Data Schema

Mỗi file đều có cùng cấu trúc:

| Column      | Data Type | Description                                            |
| ----------- | --------- | ------------------------------------------------------ |
| `user_idx`  | `int64`   | Chỉ số (index) của user, được ánh xạ từ user ID gốc.   |
| `movie_idx` | `int64`   | Chỉ số (index) của movie, được ánh xạ từ movie ID gốc. |
| `rating`    | `float64` | Giá trị rating mà user dành cho movie.                 |
| `timestamp` | `int64`   | Thời điểm user đánh giá movie (Unix timestamp).        |

## Dataset Format

* File format: **Apache Parquet**
* Data structure: **Pandas DataFrame**
* Mỗi dòng biểu diễn một user–movie interaction.
* Các giá trị `user_idx` và `movie_idx` đã được mã hóa thành các chỉ số liên tiếp bắt đầu từ `0`.
