# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Test file for proofs.py

import os
import sys
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(TEST_DIR))
MAIN_DIR = os.path.join(SRC_DIR, 'main/python')
sys.path.insert(0, MAIN_DIR)

import dicts
from proofs import data_dir

SKELETON = {
    'proof': {
        'deps': [
            {'thm': {'name': '', 'term': ''}}
        ],
        'steps': [
            {
                'step': {
                    'variables': [{'Type0': ''}],
                    'type variables': [{'Sort0': ''}],
                    'user_state': '',
                    'term': '',
                    'proven': [],
                    'action': '',
                    'constants': [{'Type0': ''}],
                    'hyps': [{'term': ''}]
                }
            }
        ],
        'isar_kwrds': [{'name': ''}],
        'methods': [{'name': ''}],
        'apply_kwrds': [{'name': ''}]
    }
}

@pytest.fixture
def temp_valid_dir(tmp_path):
    proof_file = tmp_path / "proof0.json"
    dicts.save_as_json(SKELETON, proof_file)
    return tmp_path

@pytest.fixture
def temp_invalid_dir(tmp_path):
    (tmp_path / "notaproof.txt").write_text("some content")
    return tmp_path

def test_is_valid_true(temp_valid_dir):
    assert data_dir.is_valid(temp_valid_dir)

def test_is_valid_false(temp_invalid_dir):
    assert not data_dir.is_valid(temp_invalid_dir)

def test_generate_paths(temp_valid_dir):
    paths = list(data_dir.generate_paths(temp_valid_dir))
    assert len(paths) == 1
    assert paths[0].endswith("proof0.json")

def test_get_paths(temp_valid_dir):
    paths = data_dir.get_paths(temp_valid_dir)
    assert paths == sorted(paths)

def test_find_erroneous(temp_valid_dir):
    # No error expected
    errors = data_dir.find_erroneous(temp_valid_dir)
    assert errors == []

def test_find_erroneous_with_error(tmp_path):
    bad_json = tmp_path / "proof1.json"
    bad_json.write_text('{"bad": ')  # malformed JSON
    errors = data_dir.find_erroneous(tmp_path)
    assert bad_json.as_posix() in errors

def test_delete_erroneous(tmp_path):
    bad_file = tmp_path / "proof2.json"
    bad_file.write_text("{bad json:}")  # invalid
    deleted, failed = data_dir.delete_erroneous(tmp_path)
    assert bad_file.as_posix() in deleted
    assert not failed

def test_generate_from(temp_valid_dir):
    results = list(data_dir.generate_from(temp_valid_dir))
    assert isinstance(results[0], dict)
    assert results[0]["proof"]["methods"][0]["name"] == ""

def test_apply(temp_valid_dir):
    def count_keys(acc, d): return acc + len(d)
    result = data_dir.apply(count_keys, 0, temp_valid_dir)
    assert result == 1

def test_generate_dataset_paths(temp_valid_dir):
    paths = list(data_dir.generate_dataset_paths(temp_valid_dir, split="none"))
    assert len(paths) == 1

def test_group_paths_by_logic(tmp_path):
    logic_dir = tmp_path / "HOL"
    theory_dir = logic_dir / "Main"
    theory_dir.mkdir(parents=True)
    file_path = theory_dir / "proof3.json"
    file_path.write_text('{"foo": "bar"}')

    result = data_dir.group_paths_by_logic(tmp_path)
    assert "HOL" in result
    assert "Main.thy" in result["HOL"]
    assert result["HOL"]["Main.thy"][0][1].endswith("proof3.json")