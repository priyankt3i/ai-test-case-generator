
Example (Conceptual using Pickle - Manual Save/Load):

# At the start of main_app.py
import pickle
import uuid
import os
import streamlit as st

SESSION_STATE_DIR = ".session_state_files"
os.makedirs(SESSION_STATE_DIR, exist_ok=True)

# Try to get session ID from query params (simple example)
query_params = st.query_params # Use new API > 1.30
session_id_param = query_params.get("session_id", [None])[0]

if 'session_id' not in st.session_state:
    if session_id_param:
        st.session_state.session_id = session_id_param
    else:
        st.session_state.session_id = str(uuid.uuid4())
        # Update query params to include the new session ID for refresh
        st.query_params["session_id"] = st.session_state.session_id # Use new API > 1.30

session_file = os.path.join(SESSION_STATE_DIR, f"session_{st.session_state.session_id}.pkl")

# --- Load State ---
loaded_state = False
if os.path.exists(session_file) and 'state_loaded' not in st.session_state:
    try:
        with open(session_file, "rb") as f:
            loaded_data = pickle.load(f)
            # Carefully update session state - avoid overwriting essential keys
            # This needs careful handling, especially for complex objects
            for key, value in loaded_data.items():
                 # Avoid overwriting internal Streamlit keys or session_id itself
                 if key not in ['session_id', 'state_loaded', 'uploaded_file_state']: # Exclude file state
                      st.session_state[key] = value
            st.session_state.state_loaded = True # Flag to prevent reloading
            print(f"Loaded state from {session_file}") # Use log_message
            # Note: UploadedFile objects are NOT loaded here
    except Exception as e:
        print(f"Error loading state file {session_file}: {e}") # Use log_message
        # Delete corrupted file?
        # os.remove(session_file)

# --- Initialize Defaults (only if state wasn't loaded) ---
if not loaded_state:
     # Your init_session_state() logic here, but check if keys already exist
     def init_session_state():
          # Example: only set if not already loaded
          if 'some_key' not in st.session_state:
               st.session_state.some_key = "default_value"
          # ... rest of your init ...
     init_session_state()


# --- App Logic ---
# ... your app code ...


# --- Save State (Example: triggered by a button) ---
if st.button("Save Session State"):
     try:
          # Create a dictionary of state to save
          state_to_save = {}
          for key, value in st.session_state.items():
               # Exclude non-serializable types like UploadedFile
               # Exclude internal Streamlit keys or temporary flags
               if key not in ['state_loaded', 'uploaded_file_state', 'current_file_identifier', 'refactor_request'] and not callable(value):
                    # Add more robust check for serializability if needed
                    try:
                         pickle.dumps(value) # Test picklability
                         state_to_save[key] = value
                    except (pickle.PicklingError, TypeError):
                         print(f"Skipping non-picklable key: {key}") # Use log_message

          with open(session_file, "wb") as f:
               pickle.dump(state_to_save, f)
          st.success("Session state saved!")
          print(f"Saved state to {session_file}") # Use log_message
     except Exception as e:
          st.error(f"Error saving session state: {e}")
          print(f"Error saving state file {session_file}: {e}") # Use log_message


Pros: Most flexible, handles large/complex data (with caveats for non-serializable objects).

Cons: More complex to implement, requires file system access on the server, managing session IDs and cleanup needed, security considerations with pickle. Handling UploadedFile requires saving content separately.