import logging
import unittest
from .. import log

from app.examples import get_path

class TestLog(unittest.TestCase):
    #TODO: These are really weak tests...do they need to be more robust? 
    #TODO: These are reporting NOSET for the log level...this is wrong
    def setUp(self):
        pass
    
    def test_setup_json(self):
        log.setup_logging(default_path = get_path('logging.json'))
        logger = logging.getLogger(__name__)
        self.assertEqual(logger.root.level, logging.DEBUG)
    
    def test_setup_yaml(self):
        log.setup_logging(default_path=get_path('logging.yaml'))
        logger = logging.getLogger(__name__)
        print logger.name
        self.assertEqual(logger.root.level, logging.DEBUG)

    def test_setup(self):
        log.setup_logging()
        logger = logging.getLogger(__name__)
        self.assertEqual(logger.root.level, logging.INFO)

if __name__ == '__main__':
    unittest.main()
