# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for storing configuration information


FORMATS = {
    "S": "state", 
    "SP": "state_prems", 
    "SPK": "state_prems_kwrds", 
    "SPKT": "state_prems_kwrds_terms"
}

def print_formats():
    for key, mode in FORMATS.items():
        print(f"{key}: '{mode}'")

CTX_LENGTHS = {
    "state": 512,
    "state_prems": 1024,
    "state_prems_kwrds": 1024,
    "state_prems_kwrds_terms": 1024
}

def get_context_length(mode):
    return CTX_LENGTHS.get(mode, 512)

SPLITS = {
    "TRAIN": "train", 
    "VALID": "valid", 
    "TEST": "test", 
    "NONE": "none"
}

def print_splits():
    for key, split in SPLITS.items():
        print(f"{key}: '{split}'")