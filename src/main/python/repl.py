# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Read-Eval-Print-Loop interface

import logging
from py4j.java_gateway import JavaGateway, GatewayParameters

class REPL:
    _gateway = None
    _entrypoint = None
    _minion = None
    _repl = None

    # INITIALIZATION 

    def __init__(self, logic="HOL", thy_name="Scratch.thy"):
        self.logic = logic
        self.thy_name = thy_name
        self._initialize_repl()

        log_file = 'repl_error.log'
        logging.basicConfig(filename=log_file, level=logging.ERROR,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        
    def _initialize_repl(self):
        if self._gateway is None:
            self._gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
        self._entrypoint = self._gateway.entry_point
        self._repl = self._entrypoint.get_repl(self.logic, self.thy_name)
        self._minion = self._repl.get_minion()
        print("REPL and minion initialized.")

    def get_minion(self):
        if self._minion is None:
            self._initialize_repl()
        return self._minion
    
    def go_to_end_of(self, thy_name):
        self._repl.go_to_end_of(thy_name)

    # OPERATIONS 

    def apply(self, txt):
        result = self._repl.apply(txt)
        # print(result)
        return result
    
    def reset(self):
        return self._repl.reset()
    
    def undo(self):
        return self._repl.undo()
    
    @classmethod
    def shutdown(self):
        if self._repl:
            self._repl.shutdown()
        if self._gateway:
            self._gateway.shutdown()
            print("REPL and gateway shut down.")
    
    # INFORMATION RETRIEVAL
    
    def state_string(self):
        return self._repl.state_string()
    
    def state_size(self):
        return self._repl.state_size()
    
    def latest_error(self):
        return self._repl.latest_error()
    
    def last_usr_state(self):
        return self._repl.last_usr_state()

    def show_curr(self):
        print(self._repl.state_string())

    def is_at_proof(self):
        return self._repl.is_at_proof()

    def without_subgoals(self):
        return self._repl.without_subgoals()
    
    def proof_so_far(self):
        return self._repl.proof_so_far()
    
    def last_proof(self):
        return self._repl.last_proof_of()
    
    def complete_step(self):
        self.apply("done")
        if self.latest_error():
            self.undo()
            self.apply("qed")
            if self.latest_error():
                self.apply("sorry")