import numpy as np
import pandas as pd
import torch
from torch_geometric.datasets.movie_lens_1m import MovieLens1M

torch.manual_seed(42)
np.random.seed(42)


dataset = MovieLens1M(root="./data/MovieLens1M")
data = dataset[0]

num_users_raw = data["user"].num_nodes
num_movies_raw = data["movie"].num_nodes

edge_index = data["user", "rates", "movie"].edge_index
ratings = data["user", "rates", "movie"].rating.float()
timestamps = data["user", "rates", "movie"].time.long()

# Create dataframe, schema: user movie rating timestamp
df = pd.DataFrame({
    "user": edge_index[0].numpy(),
    "movie": edge_index[1].numpy(),
    "rating": ratings.numpy(),
    "timestamp": timestamps.numpy()
})

user_interactions = df.groupby("user").size()
movie_interactions = df.groupby("movie").size()

# Convert Explicit -> Implicit Feedback
POS_THRESHOLD = 4.0

df = df[df["rating"] >= POS_THRESHOLD].copy()

print("\nImplicit Feedback")
print(f"Positive interactions: {len(df)}")

# Remove user with fewer than 5 pos interactions
MIN_INTERACTIONS = 5

user_counts = df.groupby("user").size()

valid_users = user_counts[user_counts >= MIN_INTERACTIONS].index

df = df[df["user"].isin(valid_users)].copy()

print(f"Remaining users : {df['user'].nunique()}")
print(f"Remaining movies: {df['movie'].nunique()}")

# Re-index user and movie ids
unique_users = np.sort(df["user"].unique())
unique_movies = np.sort(df["movie"].unique())

user2idx = {u: i for i, u in enumerate(unique_users)}
movie2idx = {m: i for i, m in enumerate(unique_movies)}

df["user_idx"] = df["user"].map(user2idx)
df["movie_idx"] = df["movie"].map(movie2idx)

num_users = len(unique_users)
num_items = len(unique_movies)

# SPLIT TRAIN/VAL/TEST
def split_per_user(group):

    group = group.sort_values("timestamp").copy()

    group["split"] = "train"

    group.iloc[-2, group.columns.get_loc("split")] = "val"
    group.iloc[-1, group.columns.get_loc("split")] = "test"

    return group


df = (
    df.groupby("user_idx", group_keys=False)
      .apply(split_per_user)
      .reset_index(drop=True)
)

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()
test_df = df[df["split"] == "test"].copy()

train_edges = torch.tensor(
    train_df[["user_idx", "movie_idx"]].values,
    dtype=torch.long
)

val_edges = torch.tensor(
    val_df[["user_idx", "movie_idx"]].values,
    dtype=torch.long
)

test_edges = torch.tensor(
    test_df[["user_idx", "movie_idx"]].values,
    dtype=torch.long
)

# Build bipartite graph
movie_offset = num_users

user_nodes = train_edges[:, 0]
movie_nodes = train_edges[:, 1] + movie_offset

src = torch.cat([user_nodes, movie_nodes])
dst = torch.cat([movie_nodes, user_nodes])

edge_index_train = torch.stack([src, dst], dim=0)

# Item degree: compute only from training graph
item_degree = (
    train_df
    .groupby("movie_idx")
    .size()
    .reindex(range(num_items), fill_value=0)
)

item_degree = torch.tensor(
    item_degree.values,
    dtype=torch.long
)

# Split item to head/mid/tail (20%/30%/50%
deg_np = item_degree.numpy()

p50 = np.percentile(deg_np, 50)
p80 = np.percentile(deg_np, 80)

item_popularity_group = torch.zeros(
    num_items,
    dtype=torch.long
)

item_popularity_group[deg_np >= p50] = 1
item_popularity_group[deg_np >= p80] = 2

print("\nPopularity Groups")
print(f"Tail   : {(item_popularity_group == 0).sum().item()}")
print(f"Middle : {(item_popularity_group == 1).sum().item()}")
print(f"Head   : {(item_popularity_group == 2).sum().item()}")

# Statistic
dataset_statistics = pd.DataFrame({

    "Metric": [
        "Users",
        "Items",
        "Positive interactions",
        "Train interactions",
        "Validation interactions",
        "Test interactions",
        "Average interactions / user",
        "Average interactions / item"
    ],

    "Value": [
        num_users,
        num_items,
        len(df),
        len(train_df),
        len(val_df),
        len(test_df),
        round(len(df) / num_users, 2),
        round(len(df) / num_items, 2)
    ]

})

print("\nDataset Statistics")
print(dataset_statistics)


print("\nOutput Shapes")
print(f"train_edges          : {tuple(train_edges.shape)}")
print(f"val_edges            : {tuple(val_edges.shape)}")
print(f"test_edges           : {tuple(test_edges.shape)}")
print(f"edge_index_train     : {tuple(edge_index_train.shape)}")
print(f"item_degree          : {tuple(item_degree.shape)}")
print(f"item_popularity_group: {tuple(item_popularity_group.shape)}")

# Final objects to hand over
outputs = {
    "train_edges": train_edges,
    "val_edges": val_edges,
    "test_edges": test_edges,
    "edge_index_train": edge_index_train,
    "num_users": num_users,
    "num_items": num_items,
    "item_degree": item_degree,
    "item_popularity_group": item_popularity_group,
    "dataset_statistics": dataset_statistics
}

print("\nPreprocessing completed successfully!")

