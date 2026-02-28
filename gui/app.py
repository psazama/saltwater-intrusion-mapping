import streamlit as st
import toml
import subprocess
import os
temp_dir = "./temp"

# Ensure temp directory exists
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

if "file_upload" in st.file_uploader("Upload Config File", type="toml"):
    uploaded_file = st.file_uploader("Select a TOML config file", type=["toml"])
    if uploaded_file is not None:
        # Read the uploaded file as bytes
        file_bytes = uploaded_file.getvalue()
        saved_path = os.path.join(temp_dir, uploaded_file.name)

        # Save the uploaded file to a temporary path
        with open(saved_path, "wb") as f:
            f.write(file_bytes)

        # Load the config using toml.loads
        try:
            config = toml.loads(file_bytes.decode())
            st.success("Configuration loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")

        # Run the workflow script using subprocess
        try:
            process = subprocess.Popen(["python", "examples/workflow_runner.py", "--config", saved_path],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            for line in iter(process.stdout.readline, b''):
                st.text(line.decode())
            process.stdout.close()
            return_code = process.wait()
            if return_code != 0:
                st.error(f"Workflow failed with return code: {return_code}")
        except Exception as e:
            st.error(f"An error occurred while running the workflow: {e}")
