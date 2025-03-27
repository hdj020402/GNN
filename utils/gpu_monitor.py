import subprocess
import logging
import time
import threading

class GPUMonitor:
    def __init__(self, logger: logging.Logger, interval=60) -> None:
        self.logger = logger
        self.interval = interval
        self.running = False

    def gpu_monitor(self) -> None:
        while self.running:
            try:
                result = subprocess.run(
                    [
                        'nvidia-smi',
                        '--query-gpu=utilization.gpu,memory.total,memory.used,temperature.gpu',
                        '--format=csv,noheader'
                        ], stdout=subprocess.PIPE)
                if result.returncode == 0:
                    output = result.stdout.decode('utf-8')
                    util, mem_tot, mem_used, temp = output.replace('\n', '').split(', ')
                    self.logger.info(
                        f'utilization: {util}; memory.used: {mem_used}; memory.total: {mem_tot}; temperature: {temp}'
                        )
                else:
                    self.logger.error(f"Error running nvidia-smi: {result.stderr}")
            except Exception as e:
                self.logger.error(f"Exception occurred while trying to run nvidia-smi: {e}")
            time.sleep(self.interval)

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self.gpu_monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        self.thread.join()
