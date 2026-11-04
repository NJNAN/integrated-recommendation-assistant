"""启动入口：优先自动接入模块一，失败则回退离线演示并在 stderr 告警。"""
import sys
from config.settings import offline_demo_settings, settings
from assistant.module_one import integrated_settings
from assistant.server import create_server


def prepare():
    """决定运行模式：优先自动接入模块一，失败就回退离线演示。

    例如本机没有模块一进程时，返回离线配置并把异常交给调用方告警。

    返回：
        tuple: (生效配置, 上游对象或 None, 接入异常或 None)。
    """
    upstream = None
    try:
        active, upstream = integrated_settings(settings)
        if upstream:
            upstream.start()
        return active, upstream, None
    except Exception as exc:  # 启动失败必须降级，不能拖垮本地演示
        if upstream is not None:
            try:
                upstream.close()
            except Exception:
                pass
        return offline_demo_settings(settings), None, exc


def main():
    """启动 HTTP 服务并打印地址与数据来源，退出时关服务和上游。

    例如接入成功打印“模块一真实目录”，回退时打印“离线演示”。
    """
    active, upstream, integration_error = prepare()
    if integration_error is not None:
        print(f'警告：自动接入模块一失败，已回退离线演示（{type(integration_error).__name__}: '
              f'{integration_error}）。', file=sys.stderr, flush=True)
    try:
        server = create_server(active)
    except Exception:
        if upstream:
            upstream.close()
        raise
    print(f'拾物 · 多模态推荐助手 http://{active.host}:{server.server_port}', flush=True)
    source = ('模块一真实目录 / Oblivionis v2 冻结排序 / GBP 历史价格' if upstream
              else '离线演示（自动接入失败回退，使用配置目录）' if integration_error is not None
              else '配置目录 / 离线或外部服务')
    print(f'商品及推荐：{source}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if upstream:
            upstream.close()


if __name__ == '__main__':
    main()
