brew install python@3.8 \
&& python -m pip install --upgrade pip \
&& python -m pip install virtualenv \
&& python -m virtualenv .venv --python "$(which python3.8)" \
&& source .venv/bin/activate \
&& pip install -r requirements.txt
