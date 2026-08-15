from PyQt6.QtWidgets import QMainWindow, QTabWidget

from gui.env_page import EnvPage
from gui.project_page import ProjectPage
from gui.generate_page import GeneratePage
from gui.annotate_page import AnnotatePage
from gui.config_page import ConfigPage
from gui.train_page import TrainPage
from gui.test_page import TestPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("dddd_trainer - 图形界面")
        self.resize(1100, 780)

        self.tabs = QTabWidget()
        self.env_page = EnvPage()
        self.project_page = ProjectPage()
        self.generate_page = GeneratePage()
        self.annotate_page = AnnotatePage()
        self.config_page = ConfigPage()
        self.train_page = TrainPage()
        self.test_page = TestPage()

        self.tabs.addTab(self.env_page, "环境检查")
        self.tabs.addTab(self.project_page, "项目管理")
        self.tabs.addTab(self.generate_page, "批量生成")
        self.tabs.addTab(self.annotate_page, "手动标注")
        self.tabs.addTab(self.config_page, "训练配置")
        self.tabs.addTab(self.train_page, "训练")
        self.tabs.addTab(self.test_page, "模型测试")
        self.setCentralWidget(self.tabs)

        self.project_page.projects_changed.connect(self._refresh_all_projects)
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _refresh_all_projects(self):
        self.generate_page.refresh_projects()
        self.annotate_page.refresh_projects()
        self.config_page.refresh_projects()
        self.train_page.refresh_projects()
        self.test_page.refresh_projects()

    def _sync_project(self, page):
        project = self.project_page.current_project()
        if not project:
            return
        combo = getattr(page, "project_combo", None)
        if combo is None:
            return
        if combo.findText(project) >= 0:
            combo.setCurrentText(project)

    def _on_tab_changed(self, index: int):
        page = self.tabs.widget(index)
        if page is self.env_page:
            return
        if page is self.generate_page:
            self.generate_page.refresh_projects()
            self._sync_project(self.generate_page)
        elif page is self.annotate_page:
            self.annotate_page.refresh_projects()
            self._sync_project(self.annotate_page)
        elif page is self.config_page:
            self.config_page.refresh_projects()
            self._sync_project(self.config_page)
        elif page is self.train_page:
            self.train_page.refresh_projects()
            self._sync_project(self.train_page)
        elif page is self.test_page:
            self.test_page.refresh_projects()
            self._sync_project(self.test_page)
