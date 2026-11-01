"""Start the packaged Oblivionis web system at http://127.0.0.1:5000/."""
from web_system.app import app


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
