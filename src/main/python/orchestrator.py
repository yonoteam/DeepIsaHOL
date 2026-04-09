# Maintainers:
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
#
# Orchestrator for DeepIsaHOL evaluation.
# Launches N Py4J gateways (in one JVM) and N Python eval.py workers.
#
# Usage:
#     python orchestrator.py --config path/to/config.json --workers 4

import os
import sys
import json
import time
import signal
import argparse
import subprocess

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(MAIN_DIR)))
PORTS_FILE = os.path.join(PROJECT_DIR, "ports.json")


# PORT MONITORING

def count_registered_ports():
    """Count how many ports are currently in ports.json."""
    if not os.path.exists(PORTS_FILE):
        return 0
    try:
        with open(PORTS_FILE, "r") as f:
            return len(json.load(f))
    except (json.JSONDecodeError, IOError):
        return 0


def wait_for_ports(expected, timeout=300, poll_interval=3):
    """Block until ports.json has >= expected ports registered."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        n = count_registered_ports()
        if n >= expected:
            print(f"All {expected} gateway port(s) registered.")
            return True
        print(f"Waiting for gateways... ({n}/{expected} registered)")
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Only {count_registered_ports()}/{expected} gateways registered after {timeout}s"
    )


def clean_ports_file():
    """Reset ports.json to empty."""
    with open(PORTS_FILE, "w") as f:
        json.dump({}, f)


# MAIN ORCHESTRATOR

def main():
    parser = argparse.ArgumentParser(description="DeepIsaHOL evaluation orchestrator")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--gateway-timeout", type=int, default=300,
                        help="Seconds to wait for gateways to start (default: 300)")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    eval_script = os.path.join(MAIN_DIR, "eval.py")

    if not os.path.exists(eval_script):
        print(f"Error: eval.py not found: {eval_script}")
        sys.exit(1)
    if not os.path.exists(config_path):
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    num_workers = args.workers
    children = []
    gateway_proc = None

    def shutdown_all(signum=None, frame=None):
        """Gracefully terminate all child processes."""
        print("\n[orchestrator] Shutting down all processes...")
        for proc in children:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 15
        for proc in children:
            remaining = max(0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"[orchestrator] Force-killing PID {proc.pid}")
                proc.kill()
        clean_ports_file()
        print("[orchestrator] All processes stopped. ports.json cleaned.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_all)
    signal.signal(signal.SIGTERM, shutdown_all)

    try:
        # 1. Ensure ports.json starts clean
        clean_ports_file()

        # 2. Launch multi-gateway JVM
        print(f"[orchestrator] Starting {num_workers} Py4J gateway(s) via sbt...")
        gateway_cmd = [
            "sbt",
            f"runMain isabelle_rl.Py4j_Multi_Gateway_Main {num_workers}"
        ]
        gateway_proc = subprocess.Popen(gateway_cmd, cwd=PROJECT_DIR)
        children.append(gateway_proc)

        # 3. Wait for all gateway ports to register
        wait_for_ports(num_workers, timeout=args.gateway_timeout)

        # 4. Launch Python workers
        print(f"[orchestrator] Launching {num_workers} worker(s) running eval.py...")
        workers = []
        for i in range(num_workers):
            worker_proc = subprocess.Popen(
                [sys.executable, eval_script, config_path],
                cwd=os.getcwd()
            )
            workers.append(worker_proc)
            children.append(worker_proc)
            print(f"[orchestrator] Worker {i+1} started (PID {worker_proc.pid})")

        # 5. Monitor loop: wait for all workers to finish
        while workers:
            for w in workers[:]:
                retcode = w.poll()
                if retcode is not None:
                    workers.remove(w)
                    if retcode == 0:
                        print(f"[orchestrator] Worker PID {w.pid} finished successfully.")
                    else:
                        print(f"[orchestrator] Worker PID {w.pid} exited with code {retcode}.")

            # Check gateway health
            if gateway_proc.poll() is not None:
                print("[orchestrator] Gateway JVM died unexpectedly! Shutting down.")
                shutdown_all()

            time.sleep(2)

        print("[orchestrator] All workers finished.")

    except TimeoutError as e:
        print(f"[orchestrator] {e}")
    finally:
        shutdown_all()


if __name__ == "__main__":
    main()
