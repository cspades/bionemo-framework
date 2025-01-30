# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
from pathlib import Path

import boto3
from botocore.config import Config


def parse_args():
    """Parse arguments for uploading resources to AWS."""
    parser = argparse.ArgumentParser(description="Script to upload files and directories to S3.")
    parser.add_argument(
        "--data", type=str, nargs="+", required=True, help="Paths to files or directories to upload to S3."
    )
    parser.add_argument("--s3-bucket", type=str, required=True, help="S3 bucket name.")
    parser.add_argument(
        "--s3-keypath",
        type=str,
        required=True,
        default=None,
        help="Prefix directory path where the files or directories will be uploaded on S3.",
    )
    parser.add_argument("--aws-access-id", type=str, required=False, default="team-bionemo", help="Set AWS access ID.")
    parser.add_argument("--aws-secret-key", type=str, required=False, default=None, help="Set AWS secret key.")
    parser.add_argument("--aws-region", type=str, required=False, default="us-east-1", help="Set AWS region.")
    parser.add_argument(
        "--aws-endpoint-url", type=str, required=False, default="https://pbss.s8k.io", help="Set AWS endpoint URL."
    )
    parser.add_argument(
        "--enforce-checksum",
        action="store_true",
        help="As of AWS CLI v2.23.0, the checksum is required for all objects uploaded to S3. Default: Bypass this requirement.",
    )
    return parser.parse_args()


def create_boto3_client(
    aws_endpoint_url: str | None = None,
    aws_region: str | None = None,
    aws_access_id: str | None = None,
    aws_secret_key: str | None = None,
    enforce_checksum: bool = False,
):
    """Initialize the boto3 resource client for uploading and downloading files to S3."""
    retry_config = Config(
        retries={"max_attempts": 10, "mode": "standard"},
        request_checksum_calculation="when_required" if not enforce_checksum else None,
    )
    # Initialize a resource client, which provides OOP access to AWS resources.
    return boto3.resource(
        "s3",
        endpoint_url=aws_endpoint_url,
        region_name=aws_region,
        aws_access_key_id=aws_access_id,
        aws_secret_access_key=aws_secret_key,
        config=retry_config,
    )


def upload_file(s3, filepath: Path, bucket: str, keypath: str | None = None):
    """Upload a file to S3."""
    # Resolve absolute filepath.
    file_abspath = filepath.resolve()
    # Construct key from keypath and filename.
    key = file_abspath.name
    if keypath is not None:
        key = str(Path(keypath, key))
    # Upload file to S3.
    s3.Bucket(bucket).upload_file(str(file_abspath), key)
    print(f"Uploaded {str(file_abspath)} to s3://{bucket}/{key}.")


def upload_directory(s3, dirpath: Path, bucket: str, keypath: str | None = None):
    """Upload a directory to S3."""
    # Resolve absolute directory path.
    dir_abspath = dirpath.resolve()
    # Walk through all files in directory path.
    for root, _, files in os.walk(dir_abspath):
        for file in files:
            # Construct key from keypath, directory name, relative directory path.
            # Filename is appended in upload_file from the filepath.
            key = str(Path(dir_abspath.name, os.path.relpath(root, str(dir_abspath))))
            if keypath is not None:
                key = str(Path(keypath, key))
            # Upload file to S3.
            upload_file(s3, Path(root, file).resolve(), bucket, key)


def upload(
    data: list[str],
    s3_bucket: str,
    s3_keypath: str | None = None,
    aws_endpoint_url: str | None = None,
    aws_region: str | None = None,
    aws_access_id: str | None = None,
    aws_secret_key: str | None = None,
    enforce_checksum: bool = False,
):
    """Upload a list of files or directories (recursively) to S3."""
    # Instantiate S3 client with AWS configuration.
    # If None, try to retrieve AWS credentials from ENV.
    s3 = create_boto3_client(
        aws_endpoint_url if aws_endpoint_url is not None else os.getenv("AWS_ENDPOINT_URL", None),
        aws_region if aws_region is not None else os.getenv("AWS_REGION", None),
        aws_access_id if aws_access_id is not None else os.getenv("AWS_ACCESS_KEY_ID", None),
        aws_secret_key if aws_secret_key is not None else os.getenv("AWS_SECRET_ACCESS_KEY", None),
        enforce_checksum,
    )

    # Upload data to S3.
    for dpath in data:
        # Get absolute path for dpath.
        datapath = Path(dpath).resolve()
        if datapath.is_dir():
            upload_directory(s3, datapath, s3_bucket, s3_keypath)
        elif datapath.is_file():
            upload_file(s3, datapath, s3_bucket, s3_keypath)
        else:
            print(f"{datapath} is not a file or directory. Skipping upload...")


if __name__ == "__main__":
    args = parse_args()
    upload(**vars(args))
