from flask import got_request_exception
from demo_portal.app import create_app
import traceback

app = create_app()

def on_exc(sender, exception, **extra):
    print("EXC_SIGNAL", repr(exception), flush=True)
    traceback.print_exception(type(exception), exception, exception.__traceback__)

got_request_exception.connect(on_exc, app)
app.run(host="127.0.0.1", port=5058, debug=False)
