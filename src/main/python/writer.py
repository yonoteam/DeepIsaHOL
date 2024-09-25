# Mantainers: 
# Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

from py4j.java_gateway import JavaGateway, GatewayParameters, Py4JNetworkError
import json
import os
import logging

gateway = JavaGateway()
# usage:
# entrypoint = gateway.entry_point
# reader = entrypoint.get_reader(logic, work_dir) # from data_reader.scala
# gateway.help(reader)
# gateway.shutdown() # when done

def get_theory_file_path (read_dir, read_file):
    if os.path.isfile(read_file):
        return read_file
    else:
        target_file = None
        if not read_file.endswith(".thy"):
            read_file = read_file + ".thy"
        for root, dirs, files in os.walk(read_dir):
            if read_file in files:
                target_file = os.path.join(root, read_file)
                break
        if target_file is None:
            raise FileNotFoundError(f"The file {read_file} was not found in {read_dir} or its subdirectories.")
        return target_file
    
def make_jsons (json_strs):
    proofs = []
    for json_str in json_strs:
        fixed_str = json_str.replace("\\<", "\\\\<")
        try:
            proof_data = json.loads(fixed_str)
            proofs.append(proof_data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    return proofs

def get_proofs_data (read_dir, file_name, logic = "HOL"):
    file_path = get_theory_file_path (read_dir, file_name)
    try:
        entrypoint = gateway.entry_point
        reader = entrypoint.get_reader(logic, read_dir)
        jsons = make_jsons(reader.extract(file_path))
    except Py4JNetworkError as e:
        raise ConnectionError(f"Py4J Network Error: {str(e)}")
    return jsons

def write_data(write_dir, base_name, data_list):
    counter = 0
    target_dir = os.path.join(write_dir, base_name)
    if not os.path.exists(target_dir):
      os.makedirs(target_dir)
    for data in data_list:
        file_name = f"proof{counter}.json"
        target_file_path = os.path.join(target_dir, file_name)
        if os.path.exists(target_file_path):
            raise FileExistsError(f"File {target_file_path} already exists. Aborting to avoid overwriting.")
        with open(target_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        counter += 1

def write_data_from_to (read_dir, write_dir, logic = "HOL"):
    #log_file = os.path.join(write_dir, 'error.log')
    #logging.basicConfig(filename=log_file, level=logging.ERROR, 
    #                    format='%(asctime)s - %(levelname)s - %(message)s')
    
    for current_dir, dirs, files in os.walk(read_dir):
        thy_files = [f for f in files if f.endswith(".thy")]
        if thy_files:
            # relative path from the root directory
            rel_path = os.path.relpath(current_dir, read_dir)
        
            # create corresponding structure in write_dir
            target_dir = os.path.join(write_dir, rel_path)
            os.makedirs(target_dir, exist_ok=True)

            for thy_file in thy_files:
                try:
                    file_name = thy_file.rsplit('.', 1)[0]
                    read_file_path = os.path.join(current_dir, thy_file)
                    print("Creating proofs for " + read_file_path)
                    proofs = get_proofs_data(read_dir, thy_file, logic)
                    write_data(target_dir, file_name, proofs)
                except Exception as e:
                    print(f"Failed to process {read_file_path}: {str(e)}")
                    #logging.error(f"Failed to process {read_file_path}: {str(e)}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', type=str)
    parser.add_argument('--write', type=str)
    parser.add_argument('--logic', type=str)
    args = parser.parse_args()
    write_data_from_to(args.read, args.write, args.logic)