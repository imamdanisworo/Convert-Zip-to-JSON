import streamlit as st
import json
from dbfread import DBF, DbfError

def process_dbf_files(uploaded_files):
    """
    Reads multiple DBF files, filters stock data based on code length,
    and combines them into a single dictionary.

    Args:
        uploaded_files: A list of files uploaded via Streamlit.

    Returns:
        A dictionary containing the combined and filtered stock data.
    """
    combined_data = {}
    
    # Iterate through each uploaded file
    for uploaded_file in uploaded_files:
        try:
            # Use a temporary file path or read directly from buffer if dbfread supports it
            # For simplicity and broad compatibility, let's save it temporarily
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Read the DBF file
            dbf = DBF(uploaded_file.name, load=True)
            
            # Process each record in the DBF file
            for record in dbf.records:
                # Assuming the column names are 'STK_CODE' and 'STK_CLOS' based on the file structure
                stock_code = record.get('STK_CODE', '').strip()
                stock_price = record.get('STK_CLOS', 0)

                # Filter condition: stock code length must be between 4 and 7 characters
                if 4 <= len(stock_code) <= 7:
                    # Add the valid stock data to our combined dictionary
                    # This will overwrite duplicates, keeping the value from the last file read
                    combined_data[stock_code] = stock_price
                        
        except DbfError as e:
            st.error(f"Error processing file '{uploaded_file.name}': {e}. Please ensure it is a valid DBF file.")
        except Exception as e:
            st.error(f"An unexpected error occurred with file '{uploaded_file.name}': {e}")
            
    return combined_data

# --- Streamlit App UI ---
st.set_page_config(page_title="DBF to JSON Stock Converter", layout="wide")

st.title("📈 Stock Data Converter")
st.write("Upload one or more DBF files to extract and combine stock data into a single JSON file.")
st.write("The app will only include stocks with codes that are 4 to 7 characters long.")

# File uploader allows multiple files
uploaded_files = st.file_uploader(
    "Choose your DBF files", 
    type=['dbf'], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"Processing {len(uploaded_files)} file(s)...")
    
    # Process the files
    stock_data = process_dbf_files(uploaded_files)
    
    if stock_data:
        st.success("Files processed successfully! Here is the combined JSON data.")
        
        # Convert dictionary to a formatted JSON string
        json_output = json.dumps(stock_data, indent=4)
        
        # Display the JSON in an expandable container
        with st.expander("View Combined JSON Data", expanded=True):
            st.code(json_output, language='json')
        
        # Provide a download button for the JSON file
        st.download_button(
            label="Download JSON File",
            data=json_output,
            file_name="combined_stock_data.json",
            mime="application/json"
        )
    else:
        st.warning("No valid stock data found in the uploaded files. Please check the file contents and column names ('STK_CODE', 'STK_CLOS').")

else:
    st.info("Please upload your DBF files to begin.")
