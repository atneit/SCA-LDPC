#!/bin/sh

pip freeze | grep -v simulate_rs > requirements.txt
