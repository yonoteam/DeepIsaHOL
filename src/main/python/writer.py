# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
#
# Writes proof data from input read directory to output write directory

import os
import json
import logging
from py4j.java_gateway import JavaGateway, GatewayParameters

class Writer:
    _gateway = None
    _entrypoint = None
    _minion = None
    _writer = None

    MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
    DEEPISAHOL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(MAIN_DIR)))
    PORTS_FILE = os.path.join(DEEPISAHOL_DIR, "ports.json")

    def __init__(self, read_dir, write_dir, logic="HOL"):
        self.read_dir = read_dir
        self.write_dir = write_dir
        self.logic = logic
        self.port = self._find_available_port()
        if self.port is None:
            raise RuntimeError("No available Py4j gateway found!")
        self._mark_port_unavailable(self.port)
        self._initialize_writer()

        os.makedirs(write_dir, exist_ok=True)
        log_file = os.path.join(write_dir, 'error.log')
        logging.basicConfig(filename=log_file, level=logging.ERROR, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
    
    def _find_available_port(self):
        """Read gateway_registry.json and return an available Py4j port."""
        if not os.path.exists(self.PORTS_FILE):
            return None
        
        with open(self.PORTS_FILE, "r") as f:
            try:
                ports = json.load(f)
                for port, available in ports.items():
                    if available:
                        print(f"Connecting to Py4j Gateway on port {port}")
                        return int(port)
            except json.JSONDecodeError:
                pass  # handling it above
        return None
    
    def _mark_port_unavailable(self, port):
        """Mark the selected port as unavailable."""
        if os.path.exists(self.PORTS_FILE):
            with open(self.PORTS_FILE, "r+") as f:
                try:
                    ports = json.load(f)
                    ports[str(port)] = False
                    f.seek(0)
                    json.dump(ports, f)
                    f.truncate()
                except json.JSONDecodeError:
                    pass
    
    def _mark_port_available(self, port):
        """Mark the port as available again (for shutting down)."""
        if os.path.exists(self.PORTS_FILE):
            with open(self.PORTS_FILE, "r+") as f:
                try:
                    ports = json.load(f)
                    ports[str(port)] = True
                    f.seek(0)
                    json.dump(ports, f)
                    f.truncate()
                except json.JSONDecodeError:
                    pass
    
    def _initialize_writer(self):
        try:
            if self._gateway is None:
                self._gateway = JavaGateway(
                    gateway_parameters=GatewayParameters(port=self.port, auto_convert=True)
                )
            self._entrypoint = self._gateway.entry_point
            self._writer = self._entrypoint.get_writer(self.read_dir, self.write_dir, self.logic)
            self._minion = self._writer.get_minion()
            print(f"Writer and minion initialized on port {self.port} and directory: {self.read_dir}")
        except Exception as e:
            logging.error(f"Error initializing Writer on port {self.port} and directory: {self.read_dir}: {e}")
            raise
        
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

    def shutdown_isabelle(self):
        self._writer.shutdown_isabelle()
        print("Isabelle shut down.")
    
    def shutdown_gateway(self):
        try:
            if self._writer:
                self._writer.shutdown_isabelle()
            
            self._mark_port_available(self.port)
            if self._entrypoint:
                self._entrypoint.stop()
                self._entrypoint = None

            # if self._gateway:
            #     self._gateway.shutdown()
            #     self._gateway = None
            
            print(f"Writer and Py4J gateway shut down on port {self.port}.")
        except Exception as e:
            logging.error(f"Error during gateway shutdown on port {self.port}: {e}")
    
    def isabelle_exists(self):
        return self._writer.isabelle_exists()
    
    def set_format(self, new_format):
        self._writer.set_format(new_format)

    def get_theory_file_path(self, read_file):
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

    def write_all(self):
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