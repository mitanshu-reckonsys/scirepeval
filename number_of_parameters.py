import torch
from huggingface_hub import snapshot_download

model_name = "allenai/specter2_adhoc_query"
download_dir = model_name

downloaded_dir = snapshot_download(
    repo_id=model_name,
    local_dir=download_dir,
    repo_type="model",
)

# model = torch.load(f"{downloaded_dir}/pytorch_adapter.bin", weights_only=True)
# print(model)
# # Source - https://stackoverflow.com/a/49201237
# # Posted by Fábio Perez, modified by community. See post 'Timeline' for change history
# # Retrieved 2026-05-06, License - CC BY-SA 4.0

# pytorch_total_params = sum(p.numel() for p in model.parameters())

# print(pytorch_total_params)

state_dict = torch.load(f"{downloaded_dir}/pytorch_adapter.bin", weights_only=True)
total_params = sum(t.numel() for t in state_dict.values())

print(f"Total parameters: {total_params:,}")
