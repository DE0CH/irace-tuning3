rm -rf venv
sudo DEBIAN_FRONTEND=noninteractive apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y upgrade
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install py  thon3 python3-venv python3-pip python-is-python3 r-base
python -m venv/bin/activate
venv/bin/pip install -r requirements.txt
