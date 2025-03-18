import os
import sys


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(TEST_DIR))
MAIN_DIR = os.path.join(SRC_DIR, 'main/python')
sys.path.insert(0, MAIN_DIR)

from repl import REPL

if __name__ == "__main__":
    repl = None
    try:
        repl = REPL()
        repl.apply("lemma \"\\<forall>x. P x \\<Longrightarrow> P c\"")
        repl.show_curr()
        repl.apply("apply chunche")
        print(repl.last_error())
        repl.undo()
        repl.apply("apply blast")
        repl.show_curr()
        if repl.is_at_proof():
            print("REPL in proof mode")
        else:
            print("REPL not in proof mode")
        repl.apply("done")
        print(f"\n{repl.last_proof()}")
    except Exception as e:
        raise e
    finally:
        if repl:
            repl.shutdown_gateway()
