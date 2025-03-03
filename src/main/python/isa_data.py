# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for storing configuration information


FORMATS = {
    "S": "state", 
    "SP": "state_prems", 
    "SPK": "state_prems_kwrds", 
    "SPKT": "state_prems_kwrds_terms",
    "FS": "finetune_state",
    "FSP": "finetune_state_prems", 
    "FSPK": "finetune_state_prems_kwrds", 
    "FSPKT": "finetune_state_prems_kwrds_terms",
}

CTX_LENGTHS = {
    "state": 512,
    "state_prems": 1024,
    "state_prems_kwrds": 1024,
    "state_prems_kwrds_terms": 1024,
    "finetune_state": 512,
    "finetune_state_prems": 1024,
    "finetune_state_prems_kwrds": 1024,
    "finetune_state_prems_kwrds_terms": 1024
}

def get_context_length(mode):
    return CTX_LENGTHS.get(mode)

SPLITS = {
    "TRAIN": "train", 
    "VALID": "valid", 
    "TEST": "test", 
    "NONE": "none"
}