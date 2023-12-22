
"""
cyberq_monitor

Usage:
  cyberq_monitor.py [-d] monitor [--sheet=<name>] [--tempo=<tempo>] [--reload]
                                 [--write=<outfile>] [--cooktime=<mins>] [--ip=<address>] [(-q | --quiet)]
  cyberq_monitor.py [-d] replay <infile> [--startfrom=<mins>] [--sheet=<name>] [--tempo=<tempo>]
                                 [--reload] [--write=<outfile>] [(-q | --quiet)]
  cyberq_monitor.py [-d] load <infile> [--sheet=<name>]
  cyberq_monitor.py (-h | --help)
  cyberq_monitor.py --version

Options:
  -h --help           Show this screen.
  --version           Show version.
  -d --debug          Debug mode.
  -q --quiet          Do not output readings to stdout.
  --tempo=<tempo>     Number of seconds between readings [default: 30]
  --reload            Read the initial values from the spreadsheet.
  --write=<outfile>   Write reading to outfile [default: None]
  --sheet=<name>      Name of the google sheet [default: 1fHN3vNO1q__4N4FwKcyydRmRmD6pP9DSgxEDkTbPytM]
  --startfrom=<mins>  Number of minutes to skip into replay file [default: 0]
  --cooktime=<mins>   Number of minutes that the cook is expected to last [default: 120]
  --ip=<address>      Hostname or IP address of the CyberQ [default: 10.0.1.99]
  --units=<units>     Temporature units [default: centigrade]

"""

from .cyberq_monitor import main

if __name__ == '__main__':
    import docopt
    import logging

    log = logging.getLogger(__name__)

    arguments = docopt.docopt(__doc__, version='Cyberq Monitor')
    logging.basicConfig(
        level=logging.DEBUG if arguments['--debug'] else logging.INFO)
    log.debug("Debug mode on.")

    _tempo = int(arguments['--tempo'])
    _quiet_mode = arguments['--quiet']
    _sheet_name = arguments['--sheet']
    _start_from = int(arguments['--startfrom']) * 60  # We want seconds
    _save_file = arguments['--write']
    _reload_from_sheet = arguments['--reload']
    _cooktime = int(arguments['--cooktime'])
    _address = arguments['--ip']
    _infile = arguments['<infile>']
    _units = arguments['--units']
    if arguments['replay']:
        _mode = 'replay'
    elif arguments['monitor']:
        _mode = 'monitor'
    elif arguments['load']:
        _mode = 'load'
    else:
        raise Exception("Unknown mode")

    main(_tempo, _quiet_mode, _sheet_name,
         _start_from, _save_file, _reload_from_sheet,
         _infile, _mode, _cooktime,
         _address, _units)
