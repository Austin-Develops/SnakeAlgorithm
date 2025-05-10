import multiprocessing.queues
from algorithms import *
from sharedConfig import *
from agentRunner import main_agent
import multiprocessing
import queue
import threading
import time
import json

test_agent = None
test_init = None


'''Class to be a path calculation timer'''
class Watchdog:
    def __init__(self, timeout, status, q):
        self.timeout = timeout
        self.active = threading.Event()
        self.stop_signal = False
        self.state: Board | None = None
        self.status = status
        self.out_queue = q
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_signal:
            self.active.wait()
            start_time = time.time()
            while self.active.is_set():
                if time.time() - start_time > self.timeout:
                    self.timeoutEvent()
                    self.active.clear()
                    break
                time.sleep(0.01)

    def control(self, should_start):
        if should_start:
            self.active.set()
        else:
            self.active.clear()

    def stop(self):
        self.stop_signal = True
        self.active.set()
        self.thread.join()
    
    def store(self, state):
        self.state = state
    
    def timeoutEvent(self):
        self.status.value = KILL_PROCESS
        self.out_queue.put(self.state)
        

def run_agent_test(watcher_to_tester: multiprocessing.Queue, tester_to_watcher: multiprocessing.Queue, curr_status):
    
    while True:
        data = watcher_to_tester.get()
        if data is None:
            return
        
        watchdog = Watchdog(30, curr_status, tester_to_watcher)

        agent, init_func = data
        try:
            result = main_agent(agent, init_func, return_results=True, should_render=True, watchdog=watchdog)
        except Exception as e:
            curr_status.value = KILL_PROCESS
            print('Agent crashed:')
            tester_to_watcher.put(watchdog.state)
            return

        # config.tester_to_watcher.put_nowait(result)

        curr_status.value = SUCCESSFUL_RUN
        tester_to_watcher.put(watchdog.state)


def watch_test():

    all_agents = (
        # outside_direct_agent,
        # weighted_random_agent_with_check,
        # weighted_random_agent_recursive_check,
        # hamiltonian_cycle,
        Evolved_Hamiltonian_Cycle_Solver().hamiltonian_cycle_evolved,
    )
    all_initializers = (
        # None,
        # None,
        # None,
        # hamiltonian_cycle_initializer,
        hamiltonian_cycle_initializer,
    )

    
    results = {}

    process = multiprocessing.Process(target=run_agent_test, args=(config.watcher_to_tester, config.tester_to_watcher, config.curr_status))

    process.start()

    time.sleep(0.8)

    for agent, initializer in zip(all_agents, all_initializers):
        results[agent.__name__] = {}
        print(f'Running test for', agent.__name__)
        for i in range(1, 15):
            print(f'\tTest {i}/14')
            config.watcher_to_tester.put((agent, initializer))

            start_time = time.time()

            while True:
                try:
                    final_state = config.tester_to_watcher.get(timeout=0.5)
                
                except queue.Empty:
                    time.sleep(0.2)
                    continue

                if config.curr_status.value == KILL_PROCESS:

                    end_time = time.time()
                    if process.is_alive():
                        process.terminate()
                        process.join()
                    
                    process = multiprocessing.Process(target=run_agent_test, args=(config.watcher_to_tester, config.tester_to_watcher, config.curr_status))
                    process.start()

                    datapoint = {
                        'Apples Eaten': -1 if final_state is None else len(final_state.snake.body) - 2,
                        'Time': end_time - start_time,
                        'Terminated': True
                    }
                    results[agent.__name__][f'Test {i}'] = datapoint
                    print(datapoint, flush=True)

                    time.sleep(0.3)
            
                elif config.curr_status.value == SUCCESSFUL_RUN:
                    end_time = time.time()

                    datapoint = {
                        'Apples Eaten': -1 if final_state is None else len(final_state.snake.body) - 2,
                        'Time': end_time - start_time,
                        'Terminated': False
                    }

                    results[agent.__name__][f'Test {i}'] = datapoint
                    print(datapoint, flush=True)

                else:
                    ... #shouldn't be here
                
                break
        
            with open('results.json', 'w') as results_file:
                json.dump(results, results_file, indent=4)
                results_file.flush()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    config.create_manager()
    watch_test()