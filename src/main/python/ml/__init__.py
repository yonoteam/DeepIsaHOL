



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

