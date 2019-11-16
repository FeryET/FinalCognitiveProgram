from PyQt5.QtWidgets import QApplication
from cognitive_package.view.main_window import MainWindow
import sys


def main():

    app = QApplication(sys.argv)
    view = MainWindow()
    view.show()
    sys.exit(app.exec_())


main()
