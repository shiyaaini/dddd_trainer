import os
from configs import Config
from loguru import logger


class ProjectManager:

    def __init__(self):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects")

    def create_project(self, project_name: str, single: bool = False):
        project_base_path = os.path.join(self.base_path, project_name)
        logger.info("Creating Directory... ----> {}".format(project_base_path))
        if not os.path.exists(project_base_path):
            os.mkdir(project_base_path)
            if not os.path.exists(project_base_path):
                logger.error("Directory create failed! ----> {}".format(project_base_path))
                return False
            models_path = os.path.join(project_base_path, "models")
            logger.info("Creating Directory... ----> {}".format(models_path))
            os.mkdir(models_path)

            cache_path = os.path.join(project_base_path, "cache")
            logger.info("Creating Directory... ----> {}".format(cache_path))
            os.mkdir(cache_path)

            checkpoints_path = os.path.join(project_base_path, "checkpoints")
            logger.info("Creating Directory... ----> {}".format(checkpoints_path))
            os.mkdir(checkpoints_path)

            datasets_path = os.path.join(project_base_path, "datasets")
            logger.info("Creating Directory... ----> {}".format(datasets_path))
            os.mkdir(datasets_path)

            inbox_path = os.path.join(project_base_path, "inbox")
            logger.info("Creating Directory... ----> {}".format(inbox_path))
            os.mkdir(inbox_path)

            config_path = os.path.join(os.path.join(project_base_path, "config.yaml"))
            logger.info("Creating {} Config File... ----> {}".format("CNN" if single else "CRNN", config_path))
            conf = Config(project_name)
            conf.make_config(single=single)

            logger.info("Create Project Success! ----> {}".format(project_name))
            return True
        else:
            logger.error("Directory already exists! ----> {}".format(project_base_path))
            return False

    def list_projects(self):
        if not os.path.exists(self.base_path):
            return []
        projects = []
        for name in sorted(os.listdir(self.base_path)):
            config_path = os.path.join(self.base_path, name, "config.yaml")
            if os.path.isfile(config_path):
                projects.append(name)
        return projects

    def get_project_path(self, project_name: str):
        return os.path.join(self.base_path, project_name)

    def get_datasets_path(self, project_name: str):
        return os.path.join(self.base_path, project_name, "datasets")

    def get_inbox_path(self, project_name: str):
        return os.path.join(self.base_path, project_name, "inbox")

    def get_models_path(self, project_name: str):
        return os.path.join(self.base_path, project_name, "models")

    def ensure_datasets_dir(self, project_name: str):
        datasets_path = self.get_datasets_path(project_name)
        os.makedirs(datasets_path, exist_ok=True)
        return datasets_path

    def ensure_inbox_dir(self, project_name: str):
        inbox_path = self.get_inbox_path(project_name)
        os.makedirs(inbox_path, exist_ok=True)
        return inbox_path

    def ensure_models_dir(self, project_name: str):
        models_path = self.get_models_path(project_name)
        os.makedirs(models_path, exist_ok=True)
        return models_path

    def list_onnx_models(self, project_name: str):
        models_dir = self.get_models_path(project_name)
        if not os.path.isdir(models_dir):
            return []
        files = []
        for name in sorted(os.listdir(models_dir), reverse=True):
            if name.lower().endswith(".onnx"):
                files.append(os.path.join(models_dir, name))
        return files

    def get_charsets_path(self, project_name: str):
        return os.path.join(self.get_models_path(project_name), "charsets.json")

    def reset_training(self, project_name: str):
        """
        重置训练产物：清空 cache / checkpoints / models。
        保留 datasets、inbox 图片与 config.yaml。
        返回 (ok, message, deleted_count)。
        """
        if not project_name:
            return False, "项目名为空", 0
        project_path = self.get_project_path(project_name)
        if not os.path.isdir(project_path):
            return False, f"项目不存在: {project_path}", 0

        targets = [
            os.path.join(project_path, "cache"),
            os.path.join(project_path, "checkpoints"),
            os.path.join(project_path, "models"),
        ]
        deleted = 0
        errors = []

        for directory in targets:
            os.makedirs(directory, exist_ok=True)
            for name in os.listdir(directory):
                path = os.path.join(directory, name)
                try:
                    if os.path.isfile(path) or os.path.islink(path):
                        os.remove(path)
                        deleted += 1
                    elif os.path.isdir(path):
                        # 训练产物目录下若有子目录也清掉文件
                        for root, _dirs, files in os.walk(path, topdown=False):
                            for f in files:
                                fp = os.path.join(root, f)
                                os.remove(fp)
                                deleted += 1
                            for d in _dirs:
                                os.rmdir(os.path.join(root, d))
                        os.rmdir(path)
                        deleted += 1
                except Exception as e:
                    errors.append(f"{path}: {e}")

        # CharSet 由缓存写入，重置后清空，避免旧字符集干扰
        try:
            conf = Config(project_name)
            data = conf.load_config()
            data.setdefault("Model", {})["CharSet"] = []
            conf.make_config(config_dict=data, single=bool(data.get("Model", {}).get("Word", False)))
        except Exception as e:
            errors.append(f"config CharSet: {e}")

        if errors:
            return False, "部分清理失败:\n" + "\n".join(errors), deleted
        msg = (
            f"已重置项目「{project_name}」训练状态。\n"
            f"删除文件数: {deleted}\n"
            f"已清空: cache / checkpoints / models\n"
            f"已保留: datasets、inbox 图片与 config.yaml\n"
            f"请重新「缓存数据」后再训练。"
        )
        logger.info(msg.replace("\n", " | "))
        return True, msg, deleted
