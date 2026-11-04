"""所有路径、端口与阈值集中在这里：业务代码不写死路径、端口和商品ID。"""
from dataclasses import dataclass, field, replace
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

# 模块一发布说明里公开的演示账号（不是密钥），用于切换真实目录用户。
MODULE_ONE_ACCOUNTS = (
    ('northstar', '14911.0', '北辰采购', 'demo123'),
    ('paperplane', '12748.0', '纸飞机商店', 'demo123'),
    ('greengate', '14156.0', '绿门生活馆', 'demo123'),
)


def _path(name, default):
    """从环境变量取路径，没设置就退回仓库内的默认值。

    例如 IRA_OUTPUT_DIR 可覆盖输出目录，未设置时落到仓库的 outputs/。

    参数：
        name (str): 环境变量名。
        default (str | Path): 未设置时的默认路径。

    返回：
        Path: 统一成 Path 对象，避免各处再转换。
    """
    return Path(os.getenv(name, default))


def _module_root(name):
    """定位兄弟模块的代码目录：先找同级目录，再退回仓库内的 vendor 副本。

    例如本机 ../01_recommender_system 存在就直接用，否则用 vendor/ 里的副本。

    这样别人 clone 仓库后，模块一/模块二的源码仍然可见（vendor 见仓库 vendor/README.md）；
    没有 vendor 副本时返回一个不存在的路径，调用方按“没有上游”处理，自动走离线。

    参数：
        name (str): 兄弟项目的目录名。

    返回：
        Path: 可用的模块根目录，或一个不存在的路径。
    """
    sibling = ROOT.parent / name
    return sibling if sibling.is_dir() else ROOT / 'vendor' / name


@dataclass(frozen=True)
class Settings:
    """集中存放所有路径、端口和阈值，业务代码不再写死这些常量。

    例如端口用环境变量 IRA_PORT 覆盖，默认 8080；商品目录默认 data/catalog.json。
    """
    catalog_path: Path = field(default_factory=lambda: _path('IRA_CATALOG', ROOT / 'data/catalog.json'))
    output_dir: Path = field(default_factory=lambda: _path('IRA_OUTPUT_DIR', ROOT / 'outputs'))
    web_dir: Path = ROOT / 'web'
    host: str = '127.0.0.1'
    port: int = field(default_factory=lambda: int(os.getenv('IRA_PORT', '8080')))
    top_k: int = 6
    max_top_k: int = 24
    visual_threshold: float = 0.7          # 视觉标签置信度阈值，低于它降权忽略
    session_ttl_seconds: int = 1800        # 会话 30 分钟过期
    max_sessions: int = 1000
    max_body_bytes: int = 65536
    max_media_body_bytes: int = 4 * 1024 * 1024
    upstream_timeout_seconds: float = field(default_factory=lambda: float(os.getenv('IRA_TIMEOUT', '2')))
    recommender_url: str = field(default_factory=lambda: os.getenv('IRA_RECOMMENDER_URL', '').rstrip('/'))
    recommender_username: str = field(default_factory=lambda: os.getenv('IRA_RECOMMENDER_USERNAME', ''))
    recommender_password: str = field(default_factory=lambda: os.getenv('IRA_RECOMMENDER_PASSWORD', ''), repr=False)
    recommender_accounts: tuple = field(default=(), repr=False)
    recommender_max_candidates: int = 2000
    module_one_accounts: tuple = field(default=MODULE_ONE_ACCOUNTS, repr=False)
    module_one_root: Path = field(default_factory=lambda: _path('IRA_MODULE_ONE_ROOT', _module_root('01_recommender_system')))
    module_one_port: int = field(default_factory=lambda: int(os.getenv('IRA_MODULE_ONE_PORT', '5001')))
    module_two_root: Path = field(default_factory=lambda: _path('IRA_MODULE_TWO_ROOT', _module_root('02_multimodal_ai_assistant')))
    module_two_python: str = field(default_factory=lambda: os.getenv('IRA_MODULE_TWO_PYTHON', ''))
    module_two_timeout: float = field(default_factory=lambda: float(os.getenv('IRA_MEDIA_TIMEOUT', '180')))
    module_two_text_port: int = field(default_factory=lambda: int(os.getenv('IRA_MODULE_TWO_TEXT_PORT', '8082')))
    module_two_vision_port: int = field(default_factory=lambda: int(os.getenv('IRA_MODULE_TWO_VISION_PORT', '8083')))


def offline_demo_settings(base):
    """生成离线演示配置：只用内置目录，不连模块一、不留任何账号。

    例如自动接入失败时，用它替换配置继续本地演示。

    参数：
        base (Settings): 原始配置，其余字段保持不变。

    返回：
        Settings: 清空推荐服务与账号后的新配置。
    """
    return replace(base, catalog_path=ROOT / 'data/catalog.json', recommender_url='',
                   recommender_username='', recommender_password='', recommender_accounts=(),
                   upstream_timeout_seconds=2.0)


settings = Settings()
