from typing import Protocol, Any
from webdav3.client import Client
import os
import shutil
from urllib.parse import urlparse

from utils import find_smb_mount


class RemoteStorage(Protocol):
    def send(self, local_file_name: str, remote_file_name: str):
        ...
    def retrieve(self, remote_file_name: str, local_file_name: str):
        ...
    
class WebDavRemoteStorage:
    def __init__(self, webdav_base_uri: str, username: str, passwd: str):
        self.webdav_base_uri = webdav_base_uri
        self.username = username
        self.passwd = passwd
        self.client = Client({
            'webdav_hostname': webdav_base_uri,
            'webdav_login': username,
            'webdav_password': passwd
        })
        
    def send(self, local_file_name: str, remote_file_name: str):
        """Upload a local file to the WebDAV server."""
        from webdav3.urn import Urn
        import os

        # Ensure remote path starts with a leading slash
        if not remote_file_name.startswith('/'):
            remote_file_name = '/' + remote_file_name

        urn = Urn(remote_file_name)

        # Check if parent directory exists, if not try to create it
        parent = urn.parent()
        if parent != '/' and not self.client.check(parent):
            # Try to create parent directories recursively
            parts = [p for p in parent.split('/') if p]
            current_path = ''
            for part in parts:
                current_path += '/' + part
                if not self.client.check(current_path):
                    try:
                        self.client.mkdir(current_path)
                    except:
                        pass  # Directory might already exist or we don't have permissions

        # Now upload the file
        with open(local_file_name, 'rb') as f:
            self.client.execute_request(action='upload', path=urn.quote(), data=f)

    def retrieve(self, remote_file_name: str, local_file_name: str):
        """Download a file from the WebDAV server to a local path."""
        # Ensure remote path starts with a leading slash
        if not remote_file_name.startswith('/'):
            remote_file_name = '/' + remote_file_name
        self.client.download_sync(remote_path=remote_file_name, local_path=local_file_name)

    def get_remote_file_list(self, remote_path: str = '/'):
        """
        List files and directories at the specified remote path.
        
        Args:
            remote_path: The remote directory path to list (default: '/')
            
        Returns:
            List of file/directory names in the specified path
        """
        # Ensure remote path starts with a leading slash
        if not remote_path.startswith('/'):
            remote_path = '/' + remote_path
        
        # Ensure remote path ends with a trailing slash for directory listing
        if not remote_path.endswith('/'):
            remote_path = remote_path + '/'
        
        return self.client.list(remote_path)


class SmbMountRemoteStorage:
    def __init__(self, smb_uri: str):
        """
        Initialize SMB mount remote storage.
        
        Args:
            smb_uri: SMB URI in format smb://address/sharename
            
        Raises:
            ValueError: If the SMB mount is not found
        """
        self.smb_uri = smb_uri.rstrip('/')  # Remove trailing slash if present
        
        # Find the local mount point for this SMB share
        self.mount_point = find_smb_mount(smb_uri)
        
        if self.mount_point is None:
            raise ValueError(f"SMB mount not found for URI: {smb_uri}. Please ensure the share is mounted.")
        
        print(f"SMB mount found at: {self.mount_point}")
    
    def _get_local_path(self, remote_file_name: str) -> str:
        """
        Convert a remote file name (SMB URI) to its local path on the mount.
        
        Args:
            remote_file_name: Full SMB URI like smb://address/sharename/path/to/file
            
        Returns:
            Local file path on the mount point
        """
        # Parse the URI to extract the path after the share name
        parsed = urlparse(remote_file_name)
        if parsed.scheme != 'smb':
            raise ValueError(f"Invalid SMB URI scheme: {remote_file_name}")
        
        # Get the path component and remove the sharename from it
        # Format: smb://address/sharename/path/to/file
        # We need to extract /path/to/file
        path_parts = parsed.path.lstrip('/').split('/', 1)
        if len(path_parts) < 2:
            # Just the share name, no additional path
            return self.mount_point
        
        # Get the relative path after the sharename
        relative_path = path_parts[1]
        
        # Construct the full local path
        local_path = os.path.join(self.mount_point, relative_path)
        return local_path
    
    def send(self, local_file_name: str, remote_file_name: str):
        """
        Send (copy) a local file to the SMB mount.
        
        Args:
            local_file_name: Path to the local file to send
            remote_file_name: Remote SMB URI for the destination
        """
        # Get the local path that corresponds to the remote file name
        destination_path = self._get_local_path(remote_file_name)
        
        # Optimization: If destination file already exists, assume it's the same and skip copy
        if os.path.exists(destination_path):
            print(f"File already exists at {destination_path}, skipping copy")
            return
        
        # Create parent directories if they don't exist
        destination_dir = os.path.dirname(destination_path)
        os.makedirs(destination_dir, exist_ok=True)
        
        # Copy the file to the SMB mount
        shutil.copy2(local_file_name, destination_path)
        print(f"Copied {local_file_name} to {destination_path}")
    
    def retrieve(self, remote_file_name: str, local_file_name: str):
        """
        Retrieve a file from the SMB mount by creating a symlink.
        
        Args:
            remote_file_name: Remote SMB URI of the file to retrieve
            local_file_name: Local path where the symlink should be created
        """
        # Get the local path on the mount that corresponds to the remote file
        source_path = self._get_local_path(remote_file_name)
        
        # Verify the source file exists
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Remote file not found at: {source_path}")
        
        # Create parent directories for the symlink if needed
        local_dir = os.path.dirname(local_file_name)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        
        # Remove existing symlink/file if it exists
        if os.path.lexists(local_file_name):
            os.remove(local_file_name)
        
        # Create symlink pointing to the file on the SMB mount
        os.symlink(source_path, local_file_name)
        print(f"Created symlink {local_file_name} -> {source_path}")


def factory(remote_file_type: str, info: dict[str, Any]) -> RemoteStorage | None:
    if remote_file_type == "webdav":
        server = info.get("url")
        username = info.get("username")
        password = info.get("password")
        return WebDavRemoteStorage(server, username, password)
    return None

def main():
    webdav_remote_storage = WebDavRemoteStorage("https://webdav.doodledome.org", "dmiles", "secret123")
    webdav_remote_storage.send("test_data/example.m4a", "example.m4a")
    webdav_remote_storage.retrieve("example.m4a", "/tmp/retrieved_example.m4a")
    
    # Test the list method
    print("Listing files in root directory:")
    files = webdav_remote_storage.ls("/")
    for file in files:
        print(f"  - {file}")
if __name__ == "__main__":
    main()
