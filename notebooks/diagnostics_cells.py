# Diagnostic cells for main.ipynb — Member 3
# Paste each "# %%" block as a separate cell at the END of main.ipynb (after the
# MostPopular / BPR-MF / LightGCN comparison cell) and run them in the SAME kernel.
# They reuse in-memory variables: bpr_history, lightgcn_history, bpr_results,
# lightgcn_results, main_cols, metrics_dir, project_root, train_edges,
# edge_index_train, num_users, num_items, train_user_pos_items, test_items,
# item_popularity_group, item_degree, evaluate_full_ranking, train_lightgcn, pd.

# %% [markdown]
# # Diagnostics: Loss Curves + LightGCN Layer Sweep
# 1. Loss curves — confirm the models converged within NUM_EPOCHS.
# 2. 2-layer vs 3-layer LightGCN — 3 propagation layers can over-smooth on dense
#    MovieLens-1M (embeddings collapse toward node degree), hurting accuracy and
#    amplifying popularity bias. Compare against a 2-layer variant.

# %%
import matplotlib.pyplot as plt

figures_dir = project_root / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(range(1, len(bpr_history) + 1), bpr_history, color="tab:blue", label="BPR-MF")
ax[0].set_title("BPR-MF training loss")
ax[1].plot(range(1, len(lightgcn_history) + 1), lightgcn_history, color="tab:orange",
           label="LightGCN (3 layers)")
ax[1].set_title("LightGCN training loss")
for a in ax:
    a.set_xlabel("epoch"); a.set_ylabel("mean BPR loss")
    a.grid(True, alpha=0.3); a.legend()
plt.tight_layout()
plt.savefig(figures_dir / "loss_curves.png", dpi=120, bbox_inches="tight")
plt.show()

# Convergence check: loss drop over the last 10 epochs (small => converged).
def tail_drop(history, n=10):
    return history[-n] - history[-1] if len(history) > n else None

print("BPR-MF   loss[last] =", round(bpr_history[-1], 5),
      " drop over last 10 epochs =", tail_drop(bpr_history))
print("LightGCN loss[last] =", round(lightgcn_history[-1], 5),
      " drop over last 10 epochs =", tail_drop(lightgcn_history))

# %%
# ---------- LightGCN with 2 propagation layers ----------
lightgcn2_model, lightgcn2_history = train_lightgcn(
    train_edges=train_edges,
    edge_index_train=edge_index_train,
    num_users=num_users,
    num_items=num_items,
    train_user_pos_items=train_user_pos_items,
    num_layers=2,
    device=device,
)

lightgcn2_scores = lightgcn2_model.full_sort_scores(edge_index_train).cpu()
assert lightgcn2_scores.shape == (num_users, num_items), lightgcn2_scores.shape

lightgcn2_results = evaluate_full_ranking(
    scores=lightgcn2_scores,
    train_user_pos_items=train_user_pos_items,
    test_items=test_items,
    item_group=item_popularity_group,
    item_degree=item_degree,
    k_list=[10, 20],
)

print("\nLightGCN (2 layers) results")
for name, value in lightgcn2_results.items():
    print(f"{name}: {value:.6f}")

pd.DataFrame([lightgcn2_results]).to_csv(metrics_dir / "lightgcn_2layer_results.csv", index=False)
print("\nSaved:", metrics_dir / "lightgcn_2layer_results.csv")

# %%
# ---------- Compare LightGCN 3 vs 2 layers (vs BPR-MF) ----------
layer_cmp = pd.DataFrame([
    {"Model": "BPR-MF",              **bpr_results},
    {"Model": "LightGCN (3 layers)", **lightgcn_results},
    {"Model": "LightGCN (2 layers)", **lightgcn2_results},
])[main_cols]
layer_cmp.to_csv(metrics_dir / "lightgcn_layer_comparison.csv", index=False)
print("Saved:", metrics_dir / "lightgcn_layer_comparison.csv")

# Overlay the two LightGCN loss curves.
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(lightgcn_history) + 1), lightgcn_history, label="LightGCN (3 layers)")
plt.plot(range(1, len(lightgcn2_history) + 1), lightgcn2_history, label="LightGCN (2 layers)")
plt.title("LightGCN training loss: 3 vs 2 layers")
plt.xlabel("epoch"); plt.ylabel("mean BPR loss")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(figures_dir / "lightgcn_layer_loss.png", dpi=120, bbox_inches="tight")
plt.show()

layer_cmp.round(4)
