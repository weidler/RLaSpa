import datetime
import os


class PathManager(object):
    def __init__(self):
        self.start_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.project_root = os.sep.join(os.path.normpath(os.path.abspath(__file__)).split(os.sep)[:-3])  # root_dir

    def get_subdir_under_root(self, subdir: str) -> str:
        return os.path.join(self.project_root, subdir)

    def get_ckpt_dir(self, subdir: str) -> str:
        res = os.path.join(self.project_root, 'ckpt', subdir, self.start_timestamp)
        if not os.path.exists(res):
            os.makedirs(res)
        return res

    def get_data_dir(self, subdir: str) -> str:
        data_dir = os.path.join(self.project_root, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return os.path.join(data_dir, subdir)
