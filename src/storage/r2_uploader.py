"""
R2/S3 Storage Handler for uploading PLY files
"""
import boto3
import os
from typing import Optional


class R2Uploader:
    """Upload PLY files to Cloudflare R2 or AWS S3"""
    
    def __init__(
        self,
        bucket_name: str,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ):
        """
        Initialize R2/S3 uploader.
        
        Args:
            bucket_name: S3/R2 bucket name
            access_key_id: AWS/R2 access key (or from env: AWS_ACCESS_KEY_ID)
            secret_access_key: AWS/R2 secret key (or from env: AWS_SECRET_ACCESS_KEY)
            endpoint_url: R2 endpoint (e.g., https://xxx.r2.cloudflarestorage.com)
        """
        self.bucket_name = bucket_name
        
        # Use provided credentials or environment variables
        access_key_id = access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
        secret_access_key = secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        endpoint_url = endpoint_url or os.getenv('R2_ENDPOINT_URL')
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            endpoint_url=endpoint_url,
        )
    
    def upload_ply(
        self,
        ply_bytes: bytes,
        file_name: str,
        content_type: str = 'application/octet-stream',
    ) -> str:
        """
        Upload PLY file to R2/S3.
        
        Args:
            ply_bytes: PLY file content as bytes
            file_name: Name of file in bucket (e.g., "splat_001.ply")
            content_type: MIME type
            
        Returns:
            S3 URL of uploaded file
        """
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_name,
                Body=ply_bytes,
                ContentType=content_type,
            )
            
            # Construct S3 URL
            s3_url = f"s3://{self.bucket_name}/{file_name}"
            return s3_url
            
        except Exception as e:
            print(f"Error uploading to R2/S3: {e}")
            raise
    
    def download_ply(self, file_name: str) -> bytes:
        """
        Download PLY file from R2/S3.
        
        Args:
            file_name: Name of file in bucket
            
        Returns:
            PLY file content as bytes
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_name
            )
            return response['Body'].read()
            
        except Exception as e:
            print(f"Error downloading from R2/S3: {e}")
            raise
    
    def list_files(self, prefix: str = '') -> list:
        """
        List files in bucket.
        
        Args:
            prefix: Filter by prefix (e.g., "splats/")
            
        Returns:
            List of file names
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return []
            
            return [obj['Key'] for obj in response['Contents']]
            
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def delete_ply(self, file_name: str) -> bool:
        """
        Delete PLY file from R2/S3.
        
        Args:
            file_name: Name of file to delete
            
        Returns:
            True if successful
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_name
            )
            return True
            
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False
