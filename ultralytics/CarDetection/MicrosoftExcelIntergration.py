# Function to update data in Excel file
import logging

import pandas as pd

# Function to update data in Excel file
def update_excel_with_data(app, filename, sheet_name, column_names, car_dict):
    try:
        wb = None
        for workbook in app.books:
            if workbook.fullname == filename:
                wb = workbook
                break

        if wb is None:
            wb = app.books.open(filename)

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

        # Save the workbook
        wb.save()  # Save the changes to the Excel file

    except Exception as e:
        logging.error(f"An error occurred ({type(e).__name__}): {str(e)}")