#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
VENV_DEST="$SCRIPT_DIR/../python-virtualenv"
VENV_DEST="$(realpath $VENV_DEST)"

if [ ! -d "$VENV_DEST" ]; then
    echo "Couldn't find virtual environment. Creating one..."
    python3 -m venv $VENV_DEST
fi

echo "Activating python virtual envionment..."
source $VENV_DEST/bin/activate

echo "Making sure all specified python packages are installed..."
pip install -r requirements.txt

echo "Done!"

if [[ "$0" == *setup-environment.sh ]]; then
    echo
    echo -e "This script should be run as: "
    echo -e "\tsource $0"
    echo -e "if it wasn't you need to run the following to activate the virtual environment for the current shell:"
    echo -e "\tsource $VENV_DEST/bin/activate"
fi