import pandas as pd
import os
from datetime import datetime
import uuid

DATA_DIR = "data"
GAZE_FILE = f"{DATA_DIR}/gaze_data.csv"
SESSION_FILE = f"{DATA_DIR}/sessions.csv"


class DataLogger:
    """
    Centralized module for:
    - Managing gaze data logging
    - Managing session IDs
    - Exporting data to Excel
    - Providing cleaned datasets for dashboard visualizations
    """

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)

        # Ensure gaze CSV exists
        if not os.path.exists(GAZE_FILE):
            df = pd.DataFrame(columns=["timestamp", "x", "y", "eye", "session_id"])
            df.to_csv(GAZE_FILE, index=False)

        # Ensure session CSV exists
        if not os.path.exists(SESSION_FILE):
            df = pd.DataFrame(columns=["session_id", "start_time", "end_time"])
            df.to_csv(SESSION_FILE, index=False)

        self.session_id = None

    # -----------------------------------------------------------
    # SESSION MANAGEMENT
    # -----------------------------------------------------------

    def start_session(self):
        """
        Creates a new session ID.
        """
        self.session_id = str(uuid.uuid4())
        start_time = datetime.now().isoformat()

        df = pd.DataFrame([{
            "session_id": self.session_id,
            "start_time": start_time,
            "end_time": None
        }])

        df.to_csv(SESSION_FILE, mode="a", index=False, header=False)

        return self.session_id

    def end_session(self):
        """
        Writes an end_time to the session CSV.
        """
        if self.session_id is None:
            return

        df = pd.read_csv(SESSION_FILE)
        df.loc[df["session_id"] == self.session_id, "end_time"] = datetime.now().isoformat()
        df.to_csv(SESSION_FILE, index=False)

    # -----------------------------------------------------------
    # LOGGING GAZE DATA
    # -----------------------------------------------------------

    def log_gaze(self, timestamp, x, y, eye):
        """
        Appends a single gaze record to CSV.
        """
        if self.session_id is None:
            self.start_session()

        df = pd.DataFrame([{
            "timestamp": timestamp,
            "x": int(x),
            "y": int(y),
            "eye": eye,
            "session_id": self.session_id
        }])

        df.to_csv(GAZE_FILE, mode="a", index=False, header=False)

    # -----------------------------------------------------------
    # DATA RETRIEVAL
    # -----------------------------------------------------------

    def load_gaze(self, session_id=None):
        """
        Loads gaze data, either full or session-specific.
        """
        df = pd.read_csv(GAZE_FILE)
        if session_id:
            df = df[df["session_id"] == session_id]
        return df

    def get_latest_point(self):
        """
        Returns the latest recorded gaze point.
        """
        df = pd.read_csv(GAZE_FILE)
        if df.empty:
            return None

        row = df.iloc[-1]
        return {
            "x": int(row["x"]),
            "y": int(row["y"]),
            "eye": row["eye"],
            "timestamp": row["timestamp"]
        }

    def get_all_sessions(self):
        """
        Returns all recorded sessions.
        """
        return pd.read_csv(SESSION_FILE)

    # -----------------------------------------------------------
    # EXPORTING
    # -----------------------------------------------------------

    def export_to_excel(self, output_path="output/gaze_export.xlsx"):
        """
        Exports all gaze + session data into a formatted Excel workbook.
        """
        os.makedirs("output", exist_ok=True)

        df_gaze = pd.read_csv(GAZE_FILE)
        df_session = pd.read_csv(SESSION_FILE)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_gaze.to_excel(writer, sheet_name="Gaze Data", index=False)
            df_session.to_excel(writer, sheet_name="Sessions", index=False)

        return output_path
