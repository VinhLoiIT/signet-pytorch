#!/bin/bash

echo 'Download signatures dataset'
# wget https://cedar.buffalo.edu/NIJ/data/signatures.rar

unrar x signatures.rar

mv signatures CEDAR
mv signatures.rar CEDAR