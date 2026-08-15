import os
import sys
import fire
import multiprocessing as mp

# Embed Python (runtime\) does not put cwd on sys.path; ensure project root is importable.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from loguru import logger
from utils import project_manager
from utils import cache_data
from utils import train


class App:

    def __init__(self):
        logger.info("\nHello baby~")

    def create(self, project_name: str, single: bool = False):
        logger.info("\nCreate Project ----> {}".format(project_name))
        pm = project_manager.ProjectManager()
        pm.create_project(project_name, single)

    def cache(self, project_name: str, base_path: str, search_type: str = "name"):
        logger.info("\nCaching Data ----> {}\nPath ----> {}".format(project_name, base_path))
        cache = cache_data.CacheData(project_name)
        cache.cache(base_path, search_type)
        pass

    def train(self, project_name: str):
        logger.info("\nStart Train ----> {}\n".format(project_name))
        trainer = train.Train(project_name)
        trainer.start()

    def export(self, project_name: str):
        """从最新 checkpoint 导出 onnx（训练达标但导出失败时可用）。"""
        logger.info("\nExport ONNX from latest checkpoint ----> {}\n".format(project_name))
        trainer = train.Train(project_name)
        path = trainer.export_onnx()
        logger.info("\nExport done ----> {}".format(path))


if __name__ == '__main__':
    mp.freeze_support()
    fire.Fire(App)
