# Mantainers: 
# Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz
# usage:
# entrypoint = gateway.entry_point
# reader = entrypoint.get_minion(logic, work_dir) # from data_minion.scala
# gateway.help(reader)
# gateway.shutdown() # when done

import os
import json
import logging
from py4j.java_gateway import JavaGateway, GatewayParameters, Py4JNetworkError

class Writer:
    _gateway = None
    _entrypoint = None
    _minion = None

    def __init__(self, read_dir, write_dir, logic="HOL"):
        self.read_dir = read_dir
        self.write_dir = write_dir
        self.logic = logic
        self._initialize_minion()

        os.makedirs(write_dir, exist_ok=True)
        log_file = os.path.join(write_dir, 'error.log')
        logging.basicConfig(filename=log_file, level=logging.ERROR, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
    
    def _initialize_minion(self):
        if self._gateway is None:
            self._gateway = JavaGateway()
        self._entrypoint = self._gateway.entry_point
        self._minion = self._entrypoint.get_minion(self.logic, self.read_dir)
        print(f"Initialized reader for directory: {self.read_dir}")
        
    def get_minion(self):
        if self._minion is None:
            self._initialize_minion()
        return self._minion
    
    @classmethod
    def shutdown(cls):
        if cls._gateway:
            cls._gateway.shutdown()
            cls._gateway = None
            print("Py4J gateway shut down.")

    def get_theory_file_path (self, read_file):
        if os.path.isfile(read_file):
            return read_file
        else:
            target_file = None
            if not read_file.endswith(".thy"):
                read_file += ".thy"
            for root, dirs, files in os.walk(self.read_dir):
                if read_file in files:
                    target_file = os.path.join(root, read_file)
                    break
            if target_file is None:
                raise FileNotFoundError(f"File {read_file} not found in {self.read_dir} or its subdirectories.")
            return target_file
    
    def make_jsons (self, json_strs):
        proofs = []
        for json_str in json_strs:
            fixed_str = json_str.replace("\\<", "\\\\<")
            try:
                proof_data = json.loads(fixed_str)
                proofs.append(proof_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        return proofs

    def get_proofs_data (self, file_name):
        file_path = self.get_theory_file_path (file_name)
        try:
            jsons = self.make_jsons(self._minion.extract(file_path))
        except Py4JNetworkError as e:
            logging.error(f"Py4J Network Error: {str(e)}")
            raise ConnectionError(f"Py4J Network Error: {str(e)}")
        return jsons

    def write_data(self, full_sub_dir, base_name, data_list):
        counter = 0
        target_dir = os.path.join(full_sub_dir, base_name)
        os.makedirs(target_dir, exist_ok=True)
        for data in data_list:
            file_name = f"proof{counter}.json"
            target_file_path = os.path.join(target_dir, file_name)
            if os.path.exists(target_file_path):
                logging.error(f"File {target_file_path} already exists. Aborting to avoid overwriting.")
                raise FileExistsError(f"File {target_file_path} already exists. Aborting to avoid overwriting.")
            with open(target_file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            counter += 1

    def write_all (self):
        for current_dir, dirs, files in os.walk(self.read_dir):
            thy_files = [f for f in files if f.endswith(".thy")]
            if thy_files:
                # relative path from the root directory
                rel_path = os.path.relpath(current_dir, self.read_dir)
            
                # create corresponding structure in write_dir
                target_dir = os.path.join(self.write_dir, rel_path)
                os.makedirs(target_dir, exist_ok=True)

                for thy_file in thy_files:
                    try:
                        file_name = thy_file.rsplit('.', 1)[0]
                        print("Creating proofs for " + thy_file)
                        proofs = self.get_proofs_data(thy_file)
                        self.write_data(target_dir, file_name, proofs)
                    except Exception as e:
                        logging.error(f"Failed to process {thy_file}: {str(e)}")
                        print(f"Failed to process {thy_file}: {str(e)}")
                print("Done")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', type=str)
    parser.add_argument('--write', type=str)
    parser.add_argument('--logic', type=str)
    args = parser.parse_args()
    writer = Writer(args.read, args.write, args.logic)
    writer.write_all()
    Writer.shutdown()