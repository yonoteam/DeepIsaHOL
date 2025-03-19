# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Read-Eval-Print-Loop interface

import os
import json
import logging
from py4j.java_gateway import JavaGateway, GatewayParameters

class REPL:
    _gateway = None
    _entrypoint = None
    _minion = None
    _repl = None

    MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
    DEEPISAHOL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(MAIN_DIR)))
    PORTS_FILE = os.path.join(DEEPISAHOL_DIR, "ports.json")

    # INITIALIZATION 

    def __init__(self, logic="HOL", thy_name="Scratch.thy"):
        self.logic = logic
        self.thy_name = thy_name
        self.port = self._find_available_port()
        if self.port is None:
            raise RuntimeError("No available Py4j gateway found!")
        self._mark_port_unavailable(self.port)
        self._initialize_repl()

        log_file = f'repl_error_{self.port}.log'
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
        
    def _initialize_repl(self):
        try:
            if self._gateway is None:
                self._gateway = JavaGateway(gateway_parameters=GatewayParameters(port=self.port, auto_convert=True))
            self._entrypoint = self._gateway.entry_point
            self._repl = self._entrypoint.get_repl(self.logic, self.thy_name)
            self._minion = self._repl.get_minion()
            print(f"REPL and minion initialized on port {self.port}.")
        except Exception as e:
            logging.error(f"Error initializing REPL on port {self.port}: {e}")
            raise
    
    def switch_to(self, logic, thy_name="Scratch.thy"):
        self.shutdown_isabelle()
        print(f"Switching to logic {logic} and theory {thy_name}.")
        self._repl = self._entrypoint.get_repl(logic, thy_name)
        self.logic = logic
        self.thy_name = thy_name

    def go_to_end_of(self, thy_name):
        self._repl.go_to_end_of(thy_name)

    def go_to(self, thy_name, action_text):
        self._repl.go_to(thy_name, action_text)

    def get_minion(self):
        if self._minion is None:
            self._initialize_repl()
        return self._minion


    # OPERATIONS 

    def apply(self, txt):
        result = self._repl.apply(txt)
        # print(result)
        return result
    
    def reset(self):
        return self._repl.reset()
    
    def undo(self):
        return self._repl.undo()
    
    def undoN(self, n):
        return self._repl.undoN(n)
    
    def shutdown_isabelle(self):
        self._repl.shutdown_isabelle()
        print("Isabelle shut down.")

    def shutdown_gateway(self):
        try:
            if self._repl:
                self._repl.shutdown_isabelle()
            
            self._mark_port_available(self.port)
            if self._entrypoint:
                self._entrypoint.stop()
                self._entrypoint = None

            # if self._gateway:
            #     self._gateway.shutdown()
            #     self._gateway = None
            
            print(f"REPL and gateway shut down on port {self.port}.")
        except Exception as e:
            logging.error(f"Error during gateway shutdown on port {self.port}: {e}")

    # INFORMATION RETRIEVAL
    def isabelle_exists(self):
        return self._repl.isabelle_exists()
    
    def state_string(self):
        return self._repl.state_string()
    
    def state_size(self):
        return self._repl.state_size()
    
    def last_usr_state(self):
        return self._repl.last_usr_state()
    
    def last_action(self):
        return self._repl.last_action()
    
    def last_error(self):
        return self._repl.last_error()

    def show_curr(self):
        print(self._repl.state_string())

    def is_at_proof(self):
        return self._repl.is_at_proof()

    def without_subgoals(self):
        return self._repl.without_subgoals()
    
    def proof_so_far(self):
        return self._repl.proof_so_far()
    
    def last_proof(self):
        return self._repl.last_proof()
    
    def complete_step(self):
        self.apply("done")
        if self.latest_error():
            self.undo()
            self.apply("qed")
            if self.latest_error():
                self.undo()
                self.apply("sorry (* repl-applied sorry *)")