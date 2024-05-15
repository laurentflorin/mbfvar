import os
import subprocess

def generate_docs():
    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
    subprocess.run(['make', 'html'], cwd=docs_dir)