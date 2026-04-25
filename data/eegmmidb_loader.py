"""EEG Motor Movement/Imagery Dataset pretraining loader shim.

EEGMMIDB is public on PhysioNet and replaces TUH-EEG for the default public
CLD-Trans server setup.
"""

from data.base_loader import TensorFileDataset

EEGMMIDBDataset = TensorFileDataset
