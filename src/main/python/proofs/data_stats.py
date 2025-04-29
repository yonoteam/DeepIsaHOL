# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for reading JSONs of proofs

import logging
import statistics

from collections import Counter

import str_ops
import data_dir

# DIRECTORY COMPUTATIONS

def get_approx_tokens(token_counter, proof_json):
    """To be used with proofs.apply. Accumulates the token
    count of each proof in the input directory.
    
    :param token_counter: counter to accumulate token counts
    :param proof_json: dictionary representing a proof
    :returns: updated token counter
    :rtype: collections.Counter
    """
    proof_str = str_ops.string_from(proof_json)
    proof_tokens = proof_str.split()
    for token in proof_tokens:
        token_counter[token] += 1
    return token_counter

def estimate_vocab_size(json_data_dir, coverage_threshold=0.95):
    """Estimates the optimal vocabulary size for tokenization
    of the proofs in the input directory.
    
    :param json_data_dir: path to the directory with 'proofN.json's
    :returns: the estimated optimal vocabulary size
    :rtype: int
    """
    token_counter = data_dir.apply(get_approx_tokens, Counter(), json_data_dir)
    token_freqs = token_counter.most_common()
    total_tokens = sum(token_counter.values())

    cumulative_coverage = 0.0
    optimal_vocab_size = 0
    for i, (token, freq) in enumerate(token_freqs):
        cumulative_coverage += freq / total_tokens
        if cumulative_coverage >= coverage_threshold:
            optimal_vocab_size = i + 1
            break
    
    logging.info(f"Estimated vocab size is {optimal_vocab_size}")
    return optimal_vocab_size

def accumulate_approx_split_lengths(lengths, proof, data_mode=str_ops.FORMATS["S"]):
    """
    Accumulator to be used with proofs.compute_stats. It estimates the number of tokens 
    by splitting the proof's strings of inputs (and targets) with blank spaces.

    :param lengths: pair of lists (for lengths of input-target pairs) to accumulate
    :param proof: dictionary abstracting a proof
    :param data_mode: the data format
    :rtype: tuple(list)
    """
    x_y_pairs = str_ops.inputs_targets_from(proof, data_mode=data_mode)
    lengths[0].extend(len(x.split()) for x, _ in x_y_pairs)
    lengths[1].extend(len(y.split()) for _, y in x_y_pairs)
    return lengths 

def compute_stats(accumulator, json_data_dir, **kwargs):
    """Computes the average, maximum, minimum, median, mode, and total size
    of the (input and target) tokens in the input directory's dataset using
    the estimation from the accumulator.
    
    :param accumulator: function tuple(list) -> tuple(list) with the lengths to process
    :param data_mode: the data format mode
    :returns: dictionary containing the tokenization statistics
    :rtype: dict
    """
    def get_stats(nums):
        return {
            "avg": sum(nums) / len(nums) if nums else 0,
            "max": max(nums, default=0),
            "min": min(nums, default=0),
            "median": statistics.median(nums) if nums else 0,
            "mode": statistics.mode(nums) if nums else 0
        }
    
    accumulate = lambda acc, proof: accumulator(acc, proof, **kwargs)
    x_lengths, y_lengths = data_dir.apply(accumulate, ([],[]), json_data_dir)
    x_stats, y_stats = map(get_stats, [x_lengths, y_lengths])
    return {
        "x_avg": x_stats["avg"],
        "y_avg": y_stats["avg"],
        "x_max": x_stats["max"],
        "y_max": y_stats["max"],
        "x_min": x_stats["min"],
        "y_min": y_stats["min"],
        "x_median": x_stats["median"],
        "y_median": y_stats["median"],
        "x_mode": x_stats["mode"],
        "y_mode": y_stats["mode"],
        "total_datapoints": len(x_lengths)
    }




