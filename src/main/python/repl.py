# Mantainers: 
#   Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Read-Eval-Print-Loop interface

import os
import logging
from py4j.java_gateway import JavaGateway, GatewayParameters

class REPL:
    _gateway = None
    _entrypoint = None
    _minion = None
    _repl = None

    def __init__(self, logic="HOL"):
        self.logic = logic
        self._initialize_repl()

        log_file = 'repl_error.log'
        logging.basicConfig(filename=log_file, level=logging.ERROR,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        
    def _initialize_repl(self):
        if self._gateway is None:
            self._gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
        self._entrypoint = self._gateway.entry_point
        self._repl = self._entrypoint.get_repl(self.logic)
        self._minion = self._repl.get_minion()
        print("REPL and minion initialized.")

    def get_minion(self):
        if self._minion is None:
            self._initialize_repl()
        return self._minion

    def latest_error(self):
        return self._repl.latest_error()

    def state_size(self):
        return self._repl.state_size()

    def show_curr(self):
        print(self._repl.show_curr())

    def print_state(self):
        return self._repl.print()

    def apply(self, txt):
        result = self._repl.apply(txt)
        # print(result)
        return result

    @classmethod
    def shutdown(self):
        if self._repl:
            self._repl.shutdown()
        if self._gateway:
            self._gateway.shutdown()
            print("REPL and gateway shut down.")

