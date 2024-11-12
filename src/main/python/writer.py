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
from py4j.java_gateway import JavaGateway

class Writer:
    _gateway = None
    _entrypoint = None
    _minion = None
    _writer = None

    def __init__(self, read_dir, write_dir, logic="HOL"):
        self.read_dir = read_dir
        self.write_dir = write_dir
        self.logic = logic
        self._initialize_writer()

        os.makedirs(write_dir, exist_ok=True)
        log_file = os.path.join(write_dir, 'error.log')
        logging.basicConfig(filename=log_file, level=logging.ERROR, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
    
    def _initialize_writer(self):
        if self._gateway is None:
            self._gateway = JavaGateway()
        self._entrypoint = self._gateway.entry_point
        self._writer = self._entrypoint.get_writer(self.read_dir, self.write_dir, self.logic)
        self._minion = self._writer.get_minion()
        print(f"Initialized writer and minion for directory: {self.read_dir}")
        
    def get_minion(self):
        if self._minion is None:
            self._initialize_writer()
        return self._minion
    
    def get_writer(self):
        if self._writer is None:
            self._initialize_writer()
        return self._writer
    
    # To access the methods of py4j_gateway.scala
    def get_entrypoint(self):
        if self._entrypoint is None:
            self._initialize_writer()
        return self._entrypoint
    
    # Gives information about the input JVM (i.e. Java or Scala) object
    def get_info(self, input):
        self._gateway.help(input)
    
    @classmethod
    def shutdown(cls):
        if cls._minion:
            cls._minion.isabelle().destroy()
        if cls._gateway:
            cls._gateway.shutdown()
            cls._gateway = None
            print("Py4J gateway shut down.")
    
    def set_format(self, new_format):
        self._writer.set_format(new_format)

    def get_theory_file_path (self, read_file):
        java_opt_path = self._minion.get_theory_file_path(read_file)
        if java_opt_path.isDefined():
            return java_opt_path.get().toString()
        else:
            return None

    def get_proofs_data (self, file_name):
        file_path = self.get_theory_file_path(file_name)
        java_file_path = self._entrypoint.str_to_path(file_path)
        return self._writer.get_proofs_data(java_file_path)

    def write_data(self, file_name):
        self._writer.write_data(file_name)

    def write_all (self):
        self._writer.write_all()

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