import subprocess
import streamlit as st

# Assuming this is part of a Streamlit app
# Create a spinner to show app loading status
with st.spinner('Running workflow...'):
    process = subprocess.Popen(['your-workflow-command'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()  # Reading stdout line by line
        if process.poll() is not None:
            break  # Exit the loop if the process has finished
        if output:
            st.status(output.strip())  # Update status with the last line
            # Update your scrolling log area here (e.g., append to a log list)

        # Optionally read stderr as well
        err = process.stderr.readline()  # Read error output if needed
        if err:
            st.error(err.strip())

# Preserve existing functionality by continuing to select TOMLs from examples/.