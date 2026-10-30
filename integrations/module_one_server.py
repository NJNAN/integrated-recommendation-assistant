"""Run the frozen public Flask app; keep its writable DB in module three."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1]).resolve()))
from web_system.app import create_app
app=create_app()
app.run(host='127.0.0.1',port=int(sys.argv[2]),debug=False,use_reloader=False)
