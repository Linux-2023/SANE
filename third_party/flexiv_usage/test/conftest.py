"""
conftest.py

Pytest 配置文件，定义共享的 fixtures 和配置。
"""

import pytest
import sys
import os

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--robot-sn",
        action="store",
        default=None,
        help="Robot serial number for tests"
    )


@pytest.fixture(scope="session")
def robot_sn(request):
    """获取机器人序列号"""
    sn = request.config.getoption("--robot-sn")
    if sn is None:
        try:
            from flexiv_robot import load_robot_sn_from_yaml
            sn = load_robot_sn_from_yaml(os.path.join(PROJECT_ROOT, "robot.yaml"))
        except FileNotFoundError:
            sn = "Rizon4s-063239"
    return sn


@pytest.fixture(scope="session")
def robot_config():
    """获取机器人完整配置"""
    try:
        from flexiv_robot import load_robot_config
        return load_robot_config(os.path.join(PROJECT_ROOT, "robot.yaml"))
    except FileNotFoundError:
        return {"robot_sn": "Rizon4s-063239"}

