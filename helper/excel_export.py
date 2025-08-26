# """Handles exporting the generated test cases to an Excel file."""

import io
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Tuple

# Import config and utilities
from config import EXCEL_EXPECTED_COLUMNS, EXCEL_MAX_COL_WIDTH, EXCEL_DEFAULT_COL_WIDTH, EXCEL_SHEET_NAME_MAX_LEN
from helper.utils import sanitize_filename

# Specific exception imports
from xlsxwriter.exceptions import XlsxWriterException

def _prepare_dataframe(cases: List[Dict]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Creates and standardizes a DataFrame from a list of test case dicts.
    This version expands test cases with multiple steps into multiple rows.
    """
    expanded_cases = []
    for case in cases:
        test_steps = case.get('Test Steps', [])
        expected_results = case.get('Expected Results', [])

        if isinstance(test_steps, str):
            test_steps = [test_steps]
        if isinstance(expected_results, str):
            expected_results = [expected_results]

        num_steps = max(len(test_steps), len(expected_results))

        if num_steps == 0:
            # Add a single row with empty step/result
            row = case.copy()
            row['Step #'] = ''
            row['Test Steps'] = ''
            row['Expected Results'] = ''
            expanded_cases.append(row)
            continue

        for i in range(num_steps):
            step = test_steps[i] if i < len(test_steps) else ""
            result = expected_results[i] if i < len(expected_results) else ""

            if i == 0:
                # First row contains all the main info
                row = case.copy()
                row['Step #'] = i + 1
                row['Test Steps'] = step
                row['Expected Results'] = result
                expanded_cases.append(row)
            else:
                # Subsequent rows are mostly empty
                row = {col: '' for col in EXCEL_EXPECTED_COLUMNS}
                row['Test Case ID'] = '' # Keep this empty for the merged look
                row['Step #'] = i + 1
                row['Test Steps'] = step
                row['Expected Results'] = result
                expanded_cases.append(row)

    df = pd.DataFrame(expanded_cases)
    
    # Ensure all expected columns are present
    for col in EXCEL_EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ''

    # Reorder columns to match the expected order
    df = df[EXCEL_EXPECTED_COLUMNS]

    return df, EXCEL_EXPECTED_COLUMNS


def _set_excel_column_widths(worksheet, df: pd.DataFrame, column_list: List[str]):
    """Sets column widths in the Excel sheet based on content."""
    for i, col_name in enumerate(column_list):
        try:
            # Calculate max length of data in the column (convert to string first)
            # Add 1 for potential padding, consider header length
            max_len_data = df[col_name].astype(str).fillna('').apply(len).max()
            # Ensure header length is considered
            header_len = len(str(col_name))
            # Use max of header or data length, add padding
            max_len = max(header_len, int(max_len_data) if pd.notna(max_len_data) else 0) + 2
            # Apply width limit
            width = min(max_len, EXCEL_MAX_COL_WIDTH)
            worksheet.set_column(i, i, width)
        except KeyError:
             st.warning(f"Column '{col_name}' not found in DataFrame during width calculation. Using default width.")
             worksheet.set_column(i, i, EXCEL_DEFAULT_COL_WIDTH)
        except (TypeError, ValueError) as e:
             st.warning(f"Error calculating width for column '{col_name}': {e}. Using default width.")
             worksheet.set_column(i, i, EXCEL_DEFAULT_COL_WIDTH)
        except Exception as e: # Broader fallback
             st.warning(f"Unexpected error setting width for column '{col_name}': {e}. Using default width.")
             worksheet.set_column(i, i, EXCEL_DEFAULT_COL_WIDTH)


def export_to_excel(test_cases_dict: Dict[str, Any]) -> bytes | None:
    """
    Exports the generated test cases dictionary to an Excel file in memory.

    Args:
        test_cases_dict: Dictionary with app names as keys and lists of
                         test case dicts (or error strings) as values.

    Returns:
        The Excel file content as bytes, or None if export fails.
    """
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            processed_sheet_names = {} # Track used sheet names to avoid duplicates

            for app_name, cases in test_cases_dict.items():
                # Sanitize and truncate sheet name according to Excel limits
                base_sheet_name = sanitize_filename(app_name, max_length=EXCEL_SHEET_NAME_MAX_LEN)
                sheet_name = base_sheet_name
                counter = 1
                # Handle potential duplicate sheet names after sanitization/truncation
                while sheet_name in processed_sheet_names:
                    suffix = f"_{counter}"
                    # Ensure truncation still works with the suffix
                    trunc_len = EXCEL_SHEET_NAME_MAX_LEN - len(suffix)
                    sheet_name = base_sheet_name[:trunc_len] + suffix
                    counter += 1
                processed_sheet_names[sheet_name] = app_name # Store the final name used

                # Get the corresponding worksheet object
                # Note: df.to_excel must happen first to create the sheet
                df_to_write = None
                final_cols = []

                # Prepare data based on whether generation was successful
                if isinstance(cases, list) and cases and all(isinstance(item, dict) for item in cases):
                    try:
                        df_to_write, final_cols = _prepare_dataframe(cases)
                    except Exception as df_err:
                        st.warning(f"Error preparing DataFrame for '{app_name}': {df_err}")
                        # Create an error DataFrame for this sheet
                        df_to_write = pd.DataFrame({'Error': [f"Failed to process test case data: {df_err}"]})
                        final_cols = ['Error']
                elif isinstance(cases, str): # Handle error strings from generation
                     df_to_write = pd.DataFrame({'Status': [f"Generation error for '{app_name}': {cases}"]})
                     final_cols = ['Status']
                else: # Handle empty lists or unexpected data types
                     df_to_write = pd.DataFrame({'Status': [f"No valid test cases generated or found for '{app_name}'."]})
                     final_cols = ['Status']

                # Write the DataFrame to the sheet
                df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]

                # Set column widths if we have a valid DataFrame and columns
                if df_to_write is not None and final_cols:
                    _set_excel_column_widths(worksheet, df_to_write, final_cols)

        output.seek(0)
        return output.getvalue()

    except XlsxWriterException as xe:
        st.error(f"Failed to write Excel file using xlsxwriter: {xe}")
        return None
    except Exception as e:
        # Catch potential pandas errors or other issues
        st.error(f"An unexpected error occurred during Excel export: {e}")
        return None
