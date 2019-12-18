#!/usr/bin/env python
# coding: utf-8
"""
cyberq_monitor

Usage:
  cyberq_monitor.py monitor [--sheet=<name>] [--tempo=<tempo>] [--reload]
                            [--write=<outfile>] [--cooktime=<mins>] [--ip=<address>] [(-q | --quiet)]
  cyberq_monitor.py replay <infile> [--startfrom=<mins>] [--sheet=<name>] [--tempo=<tempo>]
                            [--reload] [--write=<outfile>] [(-q | --quiet)]
  cyberq_monitor.py (-h | --help)
  cyberq_monitor.py --version

Options:
  -h --help           Show this screen.
  --version           Show version.
  -q --quiet          Do not output readings to stdout.
  --tempo=<tempo>     Number of seconds between readings [default: 30]
  --reload            Read the initial values from the spreadsheet.
  --write=<outfile>   Write reading to outfile [default: None]
  --sheet=<name>      Name of the google sheet [default: BBQ]
  --startfrom=<mins>  Number of minutes to skip into replay file [default: 0]
  --cooktime=<mins>   Number of minutes that the cook is expected to last [default: 120]
  --ip=<address>      Hostname or IP address of the CyberQ [default: 10.0.1.99]

"""
import functools
import time
import csv
import copy
from datetime import datetime, timedelta

from docopt import docopt
import pandas as pd
from lxml import objectify
import requests
import gspread
from gspread_formatting import CellFormat, NumberFormat, format_cell_range
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials

COLS = ['pit', 'pit_target', 'food1', 'food1_target', 'food2', 'food2_target', 'food3', 'food3_target', 'fan']

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(creds)

__URL_CONTROL__ = None
__URL_DATA__ = None


def set_cyberq_url(url):
    global __URL_DATA__, __URL_CONTROL__
    __URL_CONTROL__ = url
    __URL_DATA__ = f'{__URL_CONTROL__}all.xml'


def get_data_url():
    global __URL_DATA__
    return __URL_DATA__


def get_control_url():
    global __URL_CONTROL__
    return __URL_CONTROL__


def f_to_c(f):
    return int(((f - 32) / 1.8))


def mf_to_c(f):
    return f_to_c(int(float(f / 10)))


def c_to_f(c):
    return int(((float(c) * (9 / 5)) + 32))


def c_to_mf(c):
    return c_to_f(c) * 10


def fetch_target_temps_from_cyberq():
    current = get_data_from_cyberq()
    return current[2], current[4], current[6]


def set_cyberq_control(pit=None, food1=None, food2=None, food3=None, units="centigrade"):
    """Set the control values on the CyberQ"""

    # All values have to end up as fahrenheit.
    if units == "centigrade":
        pit = c_to_f(pit) if pit is not None else None
        food1 = c_to_f(food1) if food1 is not None else None
        food2 = c_to_f(food2) if food2 is not None else None
    elif units == 'fahrenheit':
        pass
    else:
        raise Exception("Unknown unit")

    # Fetch the current values
    current = get_data_from_cyberq(units='fahrenheit')

    # Overwrite any values that we want to change
    pit = pit if pit is not None else current[2]
    food1 = food1 if food1 is not None else current[4]
    food2 = food2 if food2 is not None else current[6]
    food3 = food3 if food3 is not None else current[8]

    new = {
        'EEAUTOFLUSH': 1,
        'COOK_SET': pit,
        'FOOD1_SET': food1,
        'FOOD2_SET': food2,
        'FOOD3_SET': food3,
        'COOK_NAME': 'Cook',
        '_COOK_SET': pit,
        'FOOD1_NAME': 'Food1',
        '_FOOD1_SET': food1,
        'FOOD2_NAME': 'Food2',
        '_FOOD2_SET': food2,
        'FOOD3_NAME': 'Food3',
        '_FOOD3_SET': food3,
        '_COOK_TIMER': ''}

    # Post the new values back to the form
    requests.post(get_control_url(), new)


def get_data_from_cyberq(units='centigrade'):
    readings = objectify.fromstring(requests.get(get_data_url()).text)

    raw_row = [datetime.now(),
               int(readings['COOK']['COOK_TEMP']),
               int(readings['COOK']['COOK_SET']),
               int(readings['FOOD1']['FOOD1_TEMP'] if readings['FOOD1']['FOOD1_TEMP'] != 'OPEN' else 0),
               int(readings['FOOD1']['FOOD1_SET'] if readings['FOOD1']['FOOD1_SET'] != 'OPEN' else 0),
               int(readings['FOOD2']['FOOD2_TEMP'] if readings['FOOD2']['FOOD2_TEMP'] != 'OPEN' else 0),
               int(readings['FOOD2']['FOOD2_SET'] if readings['FOOD2']['FOOD2_SET'] != 'OPEN' else 0),
               int(readings['FOOD3']['FOOD3_TEMP'] if readings['FOOD3']['FOOD3_TEMP'] != 'OPEN' else 0),
               int(readings['FOOD3']['FOOD3_SET'] if readings['FOOD3']['FOOD3_SET'] != 'OPEN' else 0),
               int(readings['OUTPUT_PERCENT'])]

    if units == 'centigrade':
        raw_row = [raw_row[0], ] + [mf_to_c(_) for _ in raw_row[1:9]] + [raw_row[9], ]
    elif units == 'fahrenheit':
        raw_row = [raw_row[0], ] + [int(float(_) / 10) for _ in raw_row[1:9]] + [raw_row[9], ]
    else:
        raise Exception("Unknown unit")

    return raw_row


def cyberq_reader():
    while True:
        yield get_data_from_cyberq()


def get_start_time(filename):
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            return datetime.combine(
                datetime.today(),
                datetime.fromtimestamp(float(row[0])).time())


def previous_file_reader(filename, start, start_from):
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)

        for row in csv_reader:
            # Remove any date element, we only care about time.
            row[0] = datetime.combine(datetime.today(),
                                      datetime.fromtimestamp(float(row[0])).time())

            # Skip any readings that are before the start_from time
            if (row[0] - start).seconds < start_from:
                continue

            # Force all readings to be numbers
            row[1:] = [int(_) for _ in row[1:]]

            yield row


def save_file_writer(reader, filename):
    """wrap a reader generator to save the recordings to a file for replay"""
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for raw_row in reader():
            # Need to write the date as a float timestamp.
            row = copy.deepcopy(raw_row)
            row[0] = row[0].timestamp()
            writer.writerow(row)
            yield raw_row


def update(sheet, df, readings, quiet_mode=False):
    (read_time, pit, pit_target, food1, food1_target, food2, food2_target, food3, food3_target, fan) = readings

    # We need to find the row in the table (and the data sheet) where this reading is to be inserted.

    # Find the index into the data sheet that is closest to the time that the reading was taken.
    index = df.index.get_loc(read_time, method='nearest')

    # Update the row in the data sheet with the new data.
    df.iloc[index] = readings[1:]

    index_date = read_time.strftime("%H:%M:%S")

    # The data frame is indexed from 0, but the Google sheet is indexes from 1 and the sheet has a title row.
    row_index = index + 2

    # Fetch the cells for the row.
    cell_list = sheet.range(f'B{row_index}:L{row_index}')

    # Set the values for the cells from our readings.
    for cell, val in zip(cell_list, readings[1:] + [index_date]):
        cell.value = val

    # Update all of the cells.
    sheet.update_cells(cell_list)

    if not quiet_mode:
        print(readings)


def fetch_target_temps_from_sheet(sheet):
    cell_list = sheet.range('B2:B4')
    return cell_list[0].value, cell_list[1].value, cell_list[2].value


def set_target_temps_in_sheet(sheet, pit, food1, food2):
    cell_list = sheet.range('C2:C4')
    cell_list[0].value = pit
    cell_list[1].value = food1
    cell_list[2].value = food2

    # Update all of the cells.
    sheet.update_cells(cell_list)


# noinspection SpellCheckingInspection,SpellCheckingInspection
def update_control_settings(sheet):
    # Read current settings from CyberQ
    pit, food1, food2 = fetch_target_temps_from_cyberq()

    # Read current values from sheet
    new_pit, new_food1, new_food2 = fetch_target_temps_from_sheet(sheet)

    if not all([_ == '' for _ in (pit, food1, food2)]):
        new_pit, new_food1, new_food2 = [int(_) for _ in (new_pit, new_food1, new_food2)]

        # If there is a difference, write new values to CyberQ
        set_cyberq_control(
            pit=new_pit if new_pit != pit else None,
            food1=new_food1 if new_food1 != food1 else None,
            food2=new_food2 if new_food2 != food2 else None,
        )

    set_target_temps_in_sheet(sheet, *fetch_target_temps_from_cyberq())


def reload(sheet):
    # Read time series data from existing sheet.
    df = get_as_dataframe(sheet)
    # Remove any unwanted columns from the end of the frame
    df = df.iloc[:, 0:len(COLS) + 1]
    # Remove any blank rows from the end of the frame
    df = df.query('time == time')

    # Convert time column to a proper datetime type
    df = df.copy()  # Shuts up a Pandas warning that is a false positive
    df['time'] = pd.to_datetime(df['time'])

    # Set the index so that it is a time series
    df = df.set_index('time')

    return df


def create_new_dataframe(start, tempo, cooktime):
    # Create new time series and write it to the sheet.
    dti = pd.date_range(
        start=start,
        end=start + timedelta(minutes=cooktime),
        freq=f'{tempo}S')

    # Create a data frame to hold the results
    df = pd.DataFrame(columns=COLS, index=dti)

    return df


def set_sheet_from_dataframe(sheet, df):
    # Clear the current sheet
    sheet.clear()

    # Write data frame to sheet
    set_with_dataframe(sheet, df, include_index=True)

    # Apply date format
    fmt = CellFormat(
        numberFormat=NumberFormat(type='TIME', pattern='HH:MM:SS')
    )

    row_count = sheet.row_count

    format_cell_range(sheet, f'A1:A{row_count}', fmt)

    # Write additional headers
    sheet.update_cell(1, 1, 'time')
    sheet.update_cell(1, len(COLS) + 2, 'realtime')


def open_sheets(sheet_name):
    doc = client.open(sheet_name)
    sheet = doc.get_worksheet(0)
    control_sheet = doc.get_worksheet(1)
    return sheet, control_sheet


def main(tempo, quiet_mode, sheet_name,
         start_from, save_file, reload_from_sheet,
         replay_mode, infile, monitor_mode,
         cooktime, address):

    set_cyberq_url(f'http://{address}/')

    if replay_mode:
        # Replay mode
        # Fetch the start time from the save file.
        start = get_start_time(infile)

        # reader function pulls the data from the selected save file
        reader = functools.partial(previous_file_reader, infile, start, start_from)

    elif monitor_mode:
        # Monitor mode
        reader = cyberq_reader

        if save_file != 'None':
            # If we have been asked to save the readings
            # wrap the reader in a file writer
            reader = functools.partial(save_file_writer, reader, save_file)

        # Start time is now but with seconds at zero.
        now = datetime.now()
        start = datetime(now.year, now.month,
                         now.day, now.hour, now.minute, 0)
    else:
        raise Exception("Should not be possible.")

    # Open the Google sheet (it must already exist).
    sheet, control_sheet = open_sheets(sheet_name)

    if reload_from_sheet:
        df = reload(sheet)
    else:
        df = create_new_dataframe(start, tempo, cooktime)
        set_sheet_from_dataframe(sheet, df)

    for i in reader():
        last = datetime.now()

        if monitor_mode:
            # If we are in live 'monitor' mode
            update_control_settings(control_sheet)

        update(sheet, df, i, quiet_mode=quiet_mode)
        delay = tempo - (datetime.now() - last).seconds - 1
        print(f'sleep for {delay}s')
        time.sleep(delay)


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Cyberq Monitor')

    _tempo = int(arguments['--tempo'])
    _quiet_mode = arguments['--quiet']
    _sheet_name = arguments['--sheet']
    _start_from = int(arguments['--startfrom']) * 60  # We want seconds
    _save_file = arguments['--write']
    _reload_from_sheet = arguments['--reload']
    _cooktime = int(arguments['--cooktime'])
    _address = arguments['--ip']
    _replay_mode = arguments['replay']
    _infile = arguments['<infile>']
    _monitor_mode = arguments['monitor']

    main(_tempo, _quiet_mode, _sheet_name,
         _start_from, _save_file, _reload_from_sheet,
         _replay_mode, _infile, _monitor_mode, _cooktime,
         _address)
