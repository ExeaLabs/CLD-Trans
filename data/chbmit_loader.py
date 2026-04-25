"""CHB-MIT loader shim.

Replace `TensorFileDataset` with the validated project-specific parser once the
scratch dataset is mounted on the server.
"""

from data.base_loader import TensorFileDataset

CHBMITDataset = TensorFileDataset
