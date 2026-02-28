import streamlit as st
import os
import subprocess

# Function to get TOML files from the examples directory
def get_toml_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.toml')]

st.title('Saltwater Intrusion Mapping')

# Get list of .toml files
examples_directory = 'examples/'
if os.path.exists(examples_directory):
    toml_files = get_toml_files(examples_directory)
else:
    toml_files = []

# Selectbox for existing TOML files
selected_toml = st.selectbox('Select a TOML file:', toml_files)

# File uploader for custom TOML
uploaded_file = st.file_uploader('Or upload your own TOML file', type='toml', key='custom_toml')

# Check if a custom file is uploaded
if uploaded_file is not None:
    config_path = uploaded_file.name  # Note: you may want to save this in a specific location
    # Optionally: save the uploaded file for later use
else:
    config_path = os.path.join(examples_directory, selected_toml) if selected_toml else None

# Button to run the workflow
if st.button('Run Workflow'):
    if config_path:
        with st.spinner('Running the workflow...'):
            # Update the command to point to the right script
            command = ['python', 'examples/workflow_runner.py', '--config', config_path]
            subprocess.run(command)
            st.success('Workflow executed successfully!')
    else:
        st.error('Please select or upload a TOML file first!')