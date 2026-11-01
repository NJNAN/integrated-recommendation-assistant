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
    return Path(os.getenv(name, default))


def _module_root(name):
    """找兄弟项目的代码：开发机优先用同级目录，否则用仓库内 vendor/ 副本。

    这样别人 clone 仓库后，模块一/模块二的源码仍然可见（vendor 见仓库 vendor/README.md）；
    没有 vendor 副本时返回一个不存在的路径，调用方按“没有上游”处理，自动走离线。
    """
    sibling = ROOT.parent / name
    return sibling if sibling.is_dir() else ROOT / 'vendor' / name


@dataclass(frozen=True)
class Settings:
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
    """回退到内置合成目录：不连模块一、不用任何账号。"""
    return replace(base, catalog_path=ROOT / 'data/catalog.json', recommender_url='',
                   recommender_username='', recommender_password='', recommender_accounts=(),
                   upstream_timeout_seconds=2.0)


settings = Settings()
