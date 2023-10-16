rm -rf venv
sudo DEBIAN_FRONTEND=noninteractive apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y upgrade
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install python3 python3-venv python3-pip python-is-python3 r-base
python -m venv venv
venv/bin/pip install -r requirements.txt
