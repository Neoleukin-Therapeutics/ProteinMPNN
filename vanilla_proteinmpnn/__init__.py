from os import path
from .protein_mpnn_utils import ProteinMPNN, tied_featurize, parse_PDB, StructureDatasetPDB
CHECKPOINT_DIRECTORY = path.join(path.dirname(path.realpath(__file__)), 'vanilla_model_weights/')
