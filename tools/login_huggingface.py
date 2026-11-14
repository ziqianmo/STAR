"""Log in without placing a Hugging Face token in shell history or chat."""

import argparse
from getpass import getpass
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    os.umask(0o077)
    credential_dir = Path(__file__).resolve().parents[1] / ".runtime/huggingface"
    credential_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    credential_dir.chmod(0o700)
    os.environ["HF_HOME"] = str(credential_dir)
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    from huggingface_hub import login

    token = getpass("Hugging Face read token (hidden): ").strip()
    if not token:
        raise SystemExit("No token entered; login cancelled.")
    try:
        login(token=token, add_to_git_credential=False)
    except Exception as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        raise SystemExit(
            f"Login failed: {type(error).__name__}, HTTP {status}."
        ) from None
    finally:
        del token
    for name in ("token", "stored_tokens"):
        path = credential_dir / name
        if path.exists():
            path.chmod(0o600)
    print("LOGIN_OK: credentials saved for the STAR dataset downloader.")


if __name__ == "__main__":
    main()
