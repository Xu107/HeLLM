import os
from datetime import datetime
import pathlib

class Logger():
    def __init__(self, path,target="",is_debug=True):
        pathlib.Path(path).mkdir(parents=True,exist_ok=True)

        self.path = path
        self.log_ = is_debug
        self.target=target
        self.logging("#"*30+"   New Logger Start   "+"#"*30)
    def logging(self, s):
        s = str(s)
        print(datetime.now().strftime('%Y-%m-%d-%H:%M:'), s)
        if self.log_:
            with open(os.path.join(self.path,f"{self.target}_log.txt"), 'a+') as f_log:
                f_log.write(str(datetime.now().strftime('%Y-%m-%d %H:%M:')) + s + '\n')
