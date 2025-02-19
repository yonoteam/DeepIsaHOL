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

SPLITS = {
    "TRAIN": "train", 
    "VALID": "valid", 
    "TEST": "test", 
    "NONE": "none"
}

def print_splits():
    for key, mode in SPLITS.items():
        print(f"{key}: '{mode}'")