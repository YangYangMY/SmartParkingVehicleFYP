# Function to update data in Excel file
import logging

import pandas as pd

# Function to update data in Excel file
def update_excel_with_car_dict(app, wb, target_filename, full_path, sheet_name, column_names, car_dict):
    try:
        wb = wb

        if wb is None:
            wb = app.books.open(target_filename)

        sheet = None
        for s in wb.sheets:
            if s.name == sheet_name:
                sheet = s
                break

        if sheet is None:
            sheet = wb.sheets.add(name=sheet_name)

        existing_data = pd.DataFrame(columns=column_names)

        # Check if the "carId" column exists in existing_data
        if "carId" not in existing_data.columns:
            existing_data["carId"] = ""

        # Iterate over the car_dict and update existing_data for each carId
        for car_id, car_data in car_dict.items():
            # Check if the carId exists in the existing_data DataFrame
            if car_id in existing_data['carId'].values:
                # Update the existing_data for the specific carId
                existing_data.loc[existing_data['carId'] == car_id, column_names[1:]] = [car_data.get(col, '') for col in column_names[1:]]
            else:
                # Append a new row if the carId is not found
                data_row = [car_id] + [car_data.get(col, 'Unknown') for col in column_names[1:]]
                new_row = pd.DataFrame([data_row], columns=existing_data.columns)
                existing_data = existing_data.dropna(how='all')
                existing_data = pd.concat([existing_data, new_row], ignore_index=True, sort=False).reset_index(drop=True)

        # Sort the DataFrame by "carId" column
        existing_data.sort_values(by=["carId"], inplace=True)

        # Use xlwings to write data to the Excel sheet
        sheet.range('A1').options(index=False, header=True).value = existing_data

        wb.save(full_path)  # Save the changes to the Excel file


    except Exception as e:
        logging.error(f"An error occurred ({type(e).__name__}): {str(e)}")


def update_excel_with_parking_lots(app, wb, target_filename, full_path, sheet_name, column_names, parking_lots):
    try:
        wb = wb

        if wb is None:
            wb = app.books.open(target_filename)

        sheet = None
        for s in wb.sheets:
            if s.name == sheet_name:
                sheet = s
                break

        if sheet is None:
            sheet = wb.sheets.add(name=sheet_name)

        existing_data = pd.DataFrame(columns=column_names)

        # Check if the "carId" column exists in existing_data
        if "parkingLot" not in existing_data.columns:
            existing_data["parkingLot"] = ""

        # Iterate over the car_dict and update existing_data for each carId
        for parking_lots_id, parking_lots_data in parking_lots.items():
            # Check if the carId exists in the existing_data DataFrame
            if parking_lots_id in existing_data['parkingLot'].values:
                # Update the existing_data for the specific carId
                existing_data.loc[existing_data['parkingLot'] == parking_lots_id, column_names[1:]] = [parking_lots_data.get(col, '') for col in column_names[1:]]
            else:
                # Append a new row if the carId is not found
                data_row = [parking_lots_id] + [parking_lots_data.get(col, 'Unknown') for col in column_names[1:]]
                new_row = pd.DataFrame([data_row], columns=existing_data.columns)
                existing_data = existing_data.dropna(how='all')
                existing_data = pd.concat([existing_data, new_row], ignore_index=True, sort=False).reset_index(drop=True)

        # Sort the DataFrame by "carId" column
        existing_data.sort_values(by=["parkingLot"], inplace=True)

        # Use xlwings to write data to the Excel sheet
        sheet.range('A1').options(index=False, header=True).value = existing_data

        wb.save(full_path)  # Save the changes to the Excel file


    except Exception as e:
        logging.error(f"An error occurred ({type(e).__name__}): {str(e)}")

def update_excel_with_double_parking_lots(app, wb, target_filename, full_path, sheet_name, column_names, double_park_lots):
    try:
        wb = wb

        if wb is None:
            wb = app.books.open(target_filename)

        sheet = None
        for s in wb.sheets:
            if s.name == sheet_name:
                sheet = s
                break

        if sheet is None:
            sheet = wb.sheets.add(name=sheet_name)

        existing_data = pd.DataFrame(columns=column_names)

        # Check if the "carId" column exists in existing_data
        if "parkingLot" not in existing_data.columns:
            existing_data["parkingLot"] = ""

        # Iterate over the car_dict and update existing_data for each carId
        for double_parking_lots_id, double_parking_lots_data in double_park_lots.items():
            # Check if the carId exists in the existing_data DataFrame
            if double_parking_lots_id in existing_data['parkingLot'].values:
                # Update the existing_data for the specific carId
                existing_data.loc[existing_data['parkingLot'] == double_parking_lots_id, column_names[1:]] = [double_parking_lots_data.get(col, '') for col in column_names[1:]]
            else:
                3
                # Append a new row if the carId is not found
                data_row = [double_parking_lots_id] + [double_parking_lots_data.get(col, 'Unknown') for col in column_names[1:]]
                new_row = pd.DataFrame([data_row], columns=existing_data.columns)
                existing_data = existing_data.dropna(how='all')
                existing_data = pd.concat([existing_data, new_row], ignore_index=True, sort=False).reset_index(drop=True)

        # Sort the DataFrame by "parkingLot" column
        existing_data.sort_values(by=["parkingLot"], inplace=True)

        # Use xlwings to write data to the Excel sheet
        sheet.range('A1').options(index=False, header=True).value = existing_data

        wb.save(full_path)  # Save the changes to the Excel file


    except Exception as e:
        logging.error(f"An error occurred ({type(e).__name__}): {str(e)}")