'''
@Author: TangZhiFeng
@Data: Do not edit
@LastEditors: TangZhiFeng
@LastEditTime: 2020-03-30 10:46:53
@Description: 
'''

import unittest   # The test framework
from models import FastText
from utils import sentance2ids



config = FastText.GenerateConfig()

class Test_utils(unittest.TestCase):
    def test_sentance2ids(self):
        result = sentance2ids('传蔡卓妍与英皇约满后跳槽金牌大风', config)
        self.assertTrue(isinstance(result, list))


if __name__ == '__main__':
    unittest.main()
