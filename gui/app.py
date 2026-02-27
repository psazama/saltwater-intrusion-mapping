import streamlit as st
import os
import toml

# Load TOML configurations
@st.cache
def load_configs(config_path):
    return toml.load(config_path)

# Function to run the workflow
def run_workflow(config):
    os.system(f'python examples/workflow_runner.py {config}')

st.title('Saltwater Intrusion Mapping GUI')

# File uploader for TOML config
config_file = st.file_uploader('Select a TOML configuration file', type='toml')
if config_file:
    configs = load_configs(config_file)
    st.write('Configurations loaded:', configs)
    if st.button('Run Workflow'):
        run_workflow(config_file.name)
        st.success('Workflow executed successfully!')

st.sidebar.title('Data Index')
# Functionality for data indexing goes here

