# Python script to install required packages and run script.py
import subprocess

def install_packages():
    packages = [
        "Flask",
        "torch",
        "transformers",
        "nltk",
        "Flask-WTF",
    ]

    for package in packages:
        subprocess.run(["pip", "install", package])

def run_script():
    subprocess.run(["python", "script.py"])

if __name__ == "__main__":
    install_packages()
    run_script()
