"""薄入口：用模块一冻结的 Flask 公开应用在本机起服务，可写库仍留在模块三。
例如传入模块一路径和 5000 端口，服务只监听 127.0.0.1。
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1]).resolve()))
from web_system.app import create_app
app=create_app()
app.run(host='127.0.0.1',port=int(sys.argv[2]),debug=False,use_reloader=False)
