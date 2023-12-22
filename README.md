CyberQ Big Green Egg Monitor
============================

python -m cyberqmonitor -h

Getting started
---------------

1. Copy the template sheet: https://docs.google.com/spreadsheets/d/1fHN3vNO1q__4N4FwKcyydRmRmD6pP9DSgxEDkTbPytM/edit#gid=570902964
2. Grab the id of the new sheet
3. Update the target cook temps on the second sheet

Start the monitor
-----------------

python -m cyberqmonitor monitor --sheet='1UfPfsFmPvLi_bEAtIvRGYimTouMylUFBlvvpM3W1o0o' \
           --write=cooks/mycook.csv --cooktime=300

Restart the monitor
-------------------

python -m cyberqmonitor monitor --reload --sheet='1UfPfsFmPvLi_bEAtIvRGYimTouMylUFBlvvpM3W1o0o' \
           --write=cooks/mycook.csv --cooktime=300
