#!/bin/bash
#
# Parse and plot actions for a given episode
#
# $1: episode
# $2: logfile
# $3: plot title
# $4: base output name
# $5: output directory
# 

awk -f parse-actions.awk -v "ep=${1}" "${2}" > "${5}/${4}_actions_ep_${1}.csv"

python plot-actions.py "${5}/${4}_actions_ep_${1}.csv" "${3}" "${5}/${4}_actions_ep_${1}.png"
