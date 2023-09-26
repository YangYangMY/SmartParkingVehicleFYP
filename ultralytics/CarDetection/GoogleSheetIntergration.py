from datetime import time

import gspread
from gspread import Cell

# Function to get the column letter for a given column number
def get_column_letter(col_num):
    result = ""
    while col_num > 0:
        col_num, remainder = divmod(col_num - 1, 26)
        result = chr(65 + remainder) + result
    return result

# Function to export car data to a Google Sheet
def ExportDatatoGSheet(car_dict_list):
    # Authenticate and open the Google Sheet outside the loop
    gc = gspread.service_account('credentials.json')
    sh = gc.open('FYP-CAR DATA SHEET')
    worksheet = sh.get_worksheet(0)

    # Define the starting row and column
    start_row = 1
    start_col = 1

    # Iterate through the list of car data dictionaries
    for car_data in car_dict_list:
        print("Exporting car data to Google Sheet...")
        print("Car Data: ", car_data)

        # Check if the car_data is a dictionary and not empty
        if isinstance(car_data, dict) and car_data:
            # Create a list of Cell objects for the current car data
            cells_to_update = []
            row = start_row

            for key, value in car_data.items():
                cell_key = get_column_letter(start_col) + str(row)
                cell = Cell(row, start_col, f"{key}: {value}")
                cells_to_update.append(cell)
                row += 1

            # Update the cells for the current car data
            worksheet.update_cells(cells_to_update)

            # Increment the starting row for the next car (if needed)
            start_row += len(car_data) + 1  # Add 1 for spacing between cars




