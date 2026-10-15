"""Download the Kaggle ASL alphabet dataset once (needs: pip install kagglehub, and Kaggle credentials)."""
import kagglehub

# Download latest version
path = kagglehub.dataset_download("grassknoted/asl-alphabet")

print("Path to dataset files:", path)
print("Next:  python -m matos extract --src kaggle=\"" + path + "/asl_alphabet_train/asl_alphabet_train\" --out shards")
