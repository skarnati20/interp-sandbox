"""
Export Script: Uploads collected benchmark trajectories and activation artifacts to Hugging Face Datasets Hub.
"""

import argparse
import os
from pathlib import Path
import sys

from huggingface_hub import HfApi, login


def parse_args():
    parser = argparse.ArgumentParser(description="Export activations dataset to Hugging Face Hub")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the local run directory to export (e.g. /workspace/data/runs/miniwob_all128_20steps)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target Hugging Face dataset repo ID (e.g. username/miniwob-qwen7b-activations)",
    )
    parser.add_argument(
        "--path_in_repo",
        type=str,
        default=None,
        help="Optional subfolder inside the dataset repository (e.g. miniwob/qwen2.5-coder-7b)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face User Access Token (with Write permission). Falls back to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the Hugging Face repository public (default is private)",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload benchmark activations and trajectory run summary",
        help="Git commit message for the upload",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_dir).resolve()

    if not data_path.exists() or not data_path.is_dir():
        print(f"Error: Data directory not found: {data_path}")
        sys.exit(1)

    # 1. Resolve HF Token
    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=True)

    api = HfApi(token=token)

    # 2. Verify Authentication
    try:
        user_info = api.whoami()
        username = user_info.get("name", "Authenticated User")
        print(f"✓ Authenticated with Hugging Face as: {username}")
    except Exception as e:
        print(f"Authentication Error: {e}")
        print("Please provide a valid token with Write access via --token or 'export HF_TOKEN=hf_...'")
        sys.exit(1)

    # 3. Create or verify Dataset repository
    is_private = not args.public
    print(f"\n[1/2] Ensuring Dataset repository exists: https://huggingface.co/datasets/{args.repo_id} (Private={is_private})...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=is_private,
        exist_ok=True,
    )

    # 4. Upload directory
    display_dest = f"{args.repo_id}/{args.path_in_repo}" if args.path_in_repo else args.repo_id
    print(f"[2/2] Uploading '{data_path}' to '{display_dest}'...")
    
    api.upload_folder(
        folder_path=str(data_path),
        path_in_repo=args.path_in_repo,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
    )

    print("\n" + "=" * 60)
    print("✓ EXPORT SUCCESSFUL!")
    print(f"Dataset URL: https://huggingface.co/datasets/{args.repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
