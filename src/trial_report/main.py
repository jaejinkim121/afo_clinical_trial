from maker import ReportMaker
import os


def get_path():
    path = os.path.dirname(os.path.abspath(__file__))
    path = path.replace('\\', '/')
    path = path[:-17]

    return path


def main():
    path = get_path()
    report = ReportMaker(path)
    report.run_gui()


if __name__ == "__main__":
    main()
