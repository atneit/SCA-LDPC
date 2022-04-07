#!/bin/sh

pip freeze | grep -v simulate-rs > requirements.txt
