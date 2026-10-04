import os
import sys

from nicegui import ui

from voice_app.ui import build_page


@ui.page('/', reconnect_timeout=60)
def index():
    build_page()


if __name__ in {'__main__', '__mp_main__'}:
    ui.run(host='127.0.0.1', port=int(os.environ.get('VOICE_APP_PORT', '2020')),
           title='語音書寫', language='zh-TW', reload=False, show='--open' in sys.argv)
