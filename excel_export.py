"""Handles exporting the generated test cases to an Excel file."""

import io
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Tuple

# Import config and utilities
from config import EXCEL_EXPECTED_COLUMNS, EXCEL_MAX_COL_WIDTH, EXCEL_DEFAULT_COL_WIDTH, EXCEL_SHEET_NAME_MAX_LEN
from utils import sanitize_filename

# Specific exception imports
from xlsxwriter.exceptions import XlsxWriterException

def _prepare_dataframe(cases: List[Dict]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Creates and standardizes a DataFrame from a list of test case dicts.

    Args:
        cases: A list of dictionaries, each representing a test case.

    Returns:
        A tuple containing:
        - pd.DataFrame: The prepared DataFrame.
        - List[str]: The final ordered list of column names in the DataFrame.
    """
    df = pd.DataFrame(cases)
    rename_dict = {}
    existing_cols_lower = {col.lower(): col for col in df.columns}
    final_cols_ordered = []

    # Ensure expected columns exist, match case-insensitively, and order them
    for expected_col in EXCEL_EXPECTED_COLUMNS:
        actual_col = existing_cols_lower.get(expected_col.lower())
        if actual_col:
            final_cols_ordered.append(actual_col)
            # If the LLM used different casing, map it to the expected casing
            if actual_col != expected_col:
                rename_dict[actual_col] = expected_col
        else:
            # Add missing expected columns with empty values
            df[expected_col] = pd.NA # Use pandas NA for consistency
            final_cols_ordered.append(expected_col)

    # Include any other columns the LLM might have added, placing them at the end
    other_cols = [
        col for col in df.columns
        if col not in final_cols_ordered and col not in rename_dict.keys() and col not in EXCEL_EXPECTED_COLUMNS
    ]
    final_column_list_before_rename = final_cols_ordered + other_cols

    # Select columns in the desired order
    df = df[final_column_list_before_rename]

    # Apply renaming to standard column names
    if rename_dict:
        df = df.rename(columns=rename_dict)
        # Get the final list of names *after* renaming
        final_column_list = [rename_dict.get(col, col) for col in final_column_list_before_rename]
    else:
        final_column_list = final_column_list_before_rename

    return df, final_column_list


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