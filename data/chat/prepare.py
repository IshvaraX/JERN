import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.jern.data import ChatDatasetPreparer

if __name__ == '__main__':
    ChatDatasetPreparer(os.path.dirname(os.path.abspath(__file__))).run()
