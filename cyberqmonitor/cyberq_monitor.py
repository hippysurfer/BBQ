#!/usr/bin/env python
# coding: utf-8
import functools
import time
import csv
import copy
from datetime import datetime, timedelta

import pandas as pd
import gspread
from gspread_formatting import CellFormat, NumberFormat, format_cell_range
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
import logging

from CyberQInterface.cyberqinterface import CyberQInterface

log = logging.getLogger(__name__)

COLS = ['pit', 'pit_target', 'food1', 'food1_target', 'food2', 'food2_target', 'food3', 'food3_target', 'fan']

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(creds)

UNIT_NAME = {
    0: 'centigrade',
    1: 'fahrenheit'
}

UNIT_CODE = {v: k for k, v in UNIT_NAME.items()}


def f_to_c(f):
    return int(((f - 32) / 1.8))


def c_to_f(c):
    return int(((float(c) * (9 / 5)) + 32))


class CyberQ(CyberQInterface):
    def __init__(self, host=None, headers=None, units='centigrade'):
        super().__init__(host=host, headers=headers)
        self._units = units
        self.set_cyberq_units()

    def convert_read_units(self, temps):
        # Convert units (CyberQ units are always in fahrenheit)
        if self._units != 'fahrenheit':
            temps = [f_to_c(_) for _ in temps]
        return temps

    def convert_write_units(self, temps):
        # Convert units (CyberQ units are always in fahrenheit)
        if self._units != 'fahrenheit':
            temps = [c_to_f(_) for _ in temps]
        return temps

    def set_cyberq_units(self):
        self.sendUpdate({'DEG_UNITS': UNIT_CODE[self._units]})

    def fetch_target_temps(self):
        config = self.getConfig()
        results = [config.COOK_SET,
                   config.FOOD1_SET,
                   config.FOOD2_SET]

        # Return a list of [pit, food1_temp, food2_temp]
        return self.convert_read_units(results)

    def set_target_temps(self, pit, food1, food2):
        update_dict = {}
        for k, v in zip(['COOK_SET', 'FOOD1_SET', 'FOOD2_SET'], [pit, food1, food2]):
            if k is not None:
                v = self.convert_write_units(temps=[v])[0]
                update_dict[k] = v

        if update_dict:
            return self.sendUpdate(update_dict)
        return None

    def get_data(self):
        # Fetch raw data
        readings = self.getAll()

        # Extract the data we want
        temperature_keys = [['COOK', 'COOK_TEMP'],
                            ['COOK', 'COOK_SET'],
                            ['FOOD1', 'FOOD1_TEMP'],
                            ['FOOD1', 'FOOD1_SET'],
                            ['FOOD2', 'FOOD2_TEMP'],
                            ['FOOD2', 'FOOD2_SET'],
                            ['FOOD3', 'FOOD3_TEMP'],
                            ['FOOD3', 'FOOD3_SET']]
        temperatures_values = [
            int(readings[prime][sub] if readings[prime][sub] != 'OPEN' else '0')
            for prime, sub in temperature_keys]

        # Convert temps to ints and scale
        temperatures_values = [_ / 10 for _ in temperatures_values]

        # Convert units (CyberQ units are always in fahrenheit)
        if self._units != 'fahrenheit':
            temperatures_values = [f_to_c(_) for _ in temperatures_values]

        # Return the list of temp values + the fan speed
        return temperatures_values + [int(readings['OUTPUT_PERCENT'])]

    def cyberq_reader(self):
        while True:
            try:
                # Yield a row of data with a timestamp
                yield [[datetime.now()] + self.get_data()]
            except Exception as e:
                print(e)
                time.sleep(1)


class DataSheet:
    def __init__(self, gsheet):
        self._gsheet = gsheet

    def load_file(self, infile):
        start = get_start_time(infile)
        saved_data = list(previous_file_reader(infile, start, 0))

        df = pd.DataFrame(data=saved_data, columns=["time"] + COLS)

        df = df.set_index('time')

        # Post data frame to sheet
        self.set_from_data_frame(df)

    def reload(self):
        # Read time series data from existing sheet.
        df = get_as_dataframe(self._gsheet)
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

    def set_from_data_frame(self, df):
        # Clear the current sheet
        self._gsheet.clear()

        # Write data frame to sheet
        set_with_dataframe(self._gsheet, df, include_index=True)

        # Apply date format
        fmt = CellFormat(
            numberFormat=NumberFormat(type='TIME', pattern='HH:MM:SS')
        )

        row_count = self._gsheet.row_count

        format_cell_range(self._gsheet, f'A1:A{row_count}', fmt)

        # Write additional headers
        self._gsheet.update_cell(1, 1, 'time')
        self._gsheet.update_cell(1, len(COLS) + 2, 'realtime')

    def update(self, df, readings, quiet_mode=False):
        (read_time, pit, pit_target, food1, food1_target, food2,
         food2_target, food3, food3_target, fan) = readings

        # We need to find the row in the table (and the data sheet) where this reading is to be inserted.

        # Find the index into the data sheet that is closest to the time that the reading was taken.
        # index = df.index.get_loc(read_time, method='nearest')
        index = df.index.get_indexer([read_time], method='nearest')

        # Update the row in the data sheet with the new data.
        df.iloc[index] = readings[1:]

        index_date = read_time.strftime("%H:%M:%S")

        # The data frame is indexed from 0, but the Google sheet is indexes from 1 and the sheet has a title row.
        row_index = index + 2

        # Fetch the cells for the row.
        cell_list = self._gsheet.range(f'B{row_index[0]}:L{row_index[0]}')

        # Set the values for the cells from our readings.
        for cell, val in zip(cell_list, readings[1:] + [index_date]):
            cell.value = val

        # Update all of the cells.
        self._gsheet.update_cells(cell_list)

        if not quiet_mode:
            print(readings)


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
            read_time = datetime.combine(
                datetime.today(),
                datetime.fromtimestamp(float(row[0])).time())

            # Skip any readings that are before the start_from time
            if (read_time - start).seconds < start_from:
                continue

            # noinspection PyTypeChecker
            row[0] = read_time

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
def update_control_settings(cyberq, sheet):
    # Read current settings from CyberQ
    pit, food1, food2 = cyberq.fetch_target_temps()

    # Read current values from sheet
    new_pit, new_food1, new_food2 = fetch_target_temps_from_sheet(sheet)

    if not all([_ == '' for _ in (pit, food1, food2)]):
        new_pit, new_food1, new_food2 = [
            int(_) for _ in (new_pit, new_food1, new_food2)]

        # If there is a difference, write new values to CyberQ
        cyberq.set_target_temps(
            pit=new_pit if new_pit != pit else None,
            food1=new_food1 if new_food1 != food1 else None,
            food2=new_food2 if new_food2 != food2 else None,
        )

    set_target_temps_in_sheet(sheet, *cyberq.fetch_target())


def create_new_dataframe(start, tempo, cooktime):
    # Create new time series and write it to the sheet.
    dti = pd.date_range(
        start=start,
        end=start + timedelta(minutes=cooktime),
        freq=f'{tempo}S')

    # Create a data frame to hold the results
    df = pd.DataFrame(columns=COLS, index=dti)

    return df


def open_sheets(gclient, sheet_name):
    log.debug(f"Opening sheet: {sheet_name}")
    doc = gclient.open_by_key(sheet_name)
    sheet = doc.get_worksheet(0)
    control_sheet = doc.get_worksheet(1)
    return DataSheet(sheet), control_sheet


# noinspection HttpUrlsUsage
def main(tempo, quiet_mode, sheet_name,
         start_from, save_file, reload_from_sheet,
         infile, mode,
         cooktime, address, units):
    # Open the Google sheet (it must already exist).
    gclient = gspread.authorize(creds)
    sheet, control_sheet = open_sheets(gclient, sheet_name)
    # noinspection PyPep8
    monitor_cb = lambda: True  # dummy for non monitor mode.

    if mode == 'load':
        sheet.load_file(infile)
        return
    elif mode == 'replay':
        # Replay mode
        # Fetch the start time from the save file.
        start = get_start_time(infile)

        # reader function pulls the data from the selected save file
        reader = functools.partial(previous_file_reader, infile, start, start_from)

    elif mode == 'monitor':
        # Monitor mode
        cyberb = CyberQ(f'http://{address}/', units=units)
        reader = cyberb.cyberq_reader

        if save_file != 'None':
            # If we have been asked to save the readings
            # wrap the reader in a file writer
            reader = functools.partial(save_file_writer, reader, save_file)

        # Start time is now but with seconds at zero.
        now = datetime.now()
        start = datetime(now.year, now.month,
                         now.day, now.hour, now.minute, 0)

        monitor_cb = functools.partial(update_control_settings, cyberb, control_sheet)

    else:
        raise Exception("Should not be possible.")

    if reload_from_sheet:
        df = sheet.reload()
    else:
        df = create_new_dataframe(start, tempo, cooktime)
        sheet.set_from_data_frame(df)

    for row in reader():
        log.debug("row={row}")
        try:
            last = datetime.now()

            if mode == 'monitor':
                # If we are in live 'monitor' mode
                monitor_cb()

            sheet.update(df, row, quiet_mode=quiet_mode)
            delay = tempo - (datetime.now() - last).seconds - 1
            log.debug(f'sleep for {delay}s')
            time.sleep(delay)
        except Exception as e:
            log.exception(e, exc_info=True)
            # It might be the sheet connection that has died, so reopen
            # use creds to create a client to interact with the Google Drive API
            gclient = gspread.authorize(creds)
            sheet, control_sheet = open_sheets(gclient, sheet_name)
