"""pytest 根配置：保证仓库根目录在 sys.path 中，railmind 可导入。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
