import logging
from flask import Flask

def create_app():
    app = Flask(__name__)
    
    logging.basicConfig(level=logging.INFO)
    
    return app