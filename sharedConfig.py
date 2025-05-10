import multiprocessing

IDLE = 0
SUCCESSFUL_RUN = 1
KILL_PROCESS = 2

class Config:
    def __init__(self):
        self.manager = None
        self.curr_status = None
        self.watcher_to_tester = None
        self.tester_to_watcher = None

    def create_manager(self):
        if self.manager is None:  # Only create manager if it's not already created
            self.manager = multiprocessing.Manager()
            self.curr_status = self.manager.Value('i', IDLE)
            self.watcher_to_tester = multiprocessing.Queue()
            self.tester_to_watcher = multiprocessing.Queue()

config = Config()