import streamlit as st
from clipboard_component import copy_component, paste_component

st.subheader("Example of Copying Text Content to Clipboard")
user_input = st.text_area(
    "Enter the content to be copied:",
    value="Type text here and click the button below to copy it to the clipboard",
    height=200
)
copy_component("Copy Button", content=user_input)

st.subheader("Clipboard Read Component")
clipboard_content = paste_component("Read Clipboard")
if clipboard_content:
    st.markdown("### Current Clipboard Content:")
    st.code(clipboard_content)
else:
    st.markdown("Click the button to read clipboard content")
