import visualdl
from visualdl.server import app

visualdl.server.app.run(logdir="./log",
                        host="127.0.0.1",
                        port=8080,
                        cache_timeout=20,
                        language=None,
                        public_path=None,
                        api_only=False,
                        open_browser=False)

