import os
import time
import sqlite3
import threading
from datetime import datetime
import pandas as pd
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()


class DatabaseManager:
    """
    Connects to MySQL (`vision_platform`) using mysql-connector-python.
    Automatically falls back to a cloud-compatible SQL engine with the exact
    same `detection_logs` schema when hosted on Streamlit Community Cloud.
    """

    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.user = os.getenv("DB_USER", "root")
        self.password = os.getenv("DB_PASSWORD", "")
        self.database = os.getenv("DB_NAME", "vision_platform")
        self.port = int(os.getenv("DB_PORT", 3306))

        self._lock = threading.Lock()
        self._last_logged_time = {}
        self.use_mysql = False
        self.sqlite_path = "vision_platform.db"
        self.init_db()

    def _get_mysql_connection(self, include_db=True):
        kwargs = {
            "host": self.host,
            "user": self.user,
            "password": self.password,
            "port": self.port,
            "connection_timeout": 3,
        }
        if include_db:
            kwargs["database"] = self.database
        return mysql.connector.connect(**kwargs)

    def init_db(self) -> bool:
        # 1. Try local/remote MySQL server first
        try:
            conn = self._get_mysql_connection(include_db=False)
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.database}`;")
            cursor.execute(f"USE `{self.database}`;")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_logs (
                    log_id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    object_class VARCHAR(100) NOT NULL,
                    confidence FLOAT NOT NULL,
                    bbox_x INT NOT NULL,
                    bbox_y INT NOT NULL,
                    bbox_w INT NOT NULL,
                    bbox_h INT NOT NULL
                );
                """
            )
            conn.commit()
            cursor.close()
            conn.close()
            self.use_mysql = True
            return True
        except Error:
            # 2. Cloud Website Fallback (mirrors exact MySQL schema)
            self.use_mysql = False
            conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    object_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x INTEGER NOT NULL,
                    bbox_y INTEGER NOT NULL,
                    bbox_w INTEGER NOT NULL,
                    bbox_h INTEGER NOT NULL
                );
                """
            )
            conn.commit()
            conn.close()
            return True

    def insert_detection(
        self,
        object_class: str,
        confidence: float,
        bbox_x: int,
        bbox_y: int,
        bbox_w: int,
        bbox_h: int,
        cooldown_seconds: float = 1.0,
    ) -> bool:
        now = time.time()
        with self._lock:
            if (now - self._last_logged_time.get(object_class, 0.0)) < cooldown_seconds:
                return False
            self._last_logged_time[object_class] = now

        if self.use_mysql:
            try:
                conn = self._get_mysql_connection(include_db=True)
                cursor = conn.cursor()
                query = """
                    INSERT INTO detection_logs
                    (object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(
                    query,
                    (
                        str(object_class),
                        round(float(confidence), 4),
                        int(bbox_x),
                        int(bbox_y),
                        int(bbox_w),
                        int(bbox_h),
                    ),
                )
                conn.commit()
                cursor.close()
                conn.close()
                return True
            except Error:
                self.use_mysql = False

        # Cloud SQL execution
        conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        cursor = conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO detection_logs
            (timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                str(object_class),
                round(float(confidence), 4),
                int(bbox_x),
                int(bbox_y),
                int(bbox_w),
                int(bbox_h),
            ),
        )
        conn.commit()
        conn.close()
        return True

    def fetch_recent_logs(self, limit: int = 20) -> pd.DataFrame:
        cols = [
            "log_id",
            "timestamp",
            "object_class",
            "confidence",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
        ]
        if self.use_mysql:
            try:
                conn = self._get_mysql_connection(include_db=True)
                cursor = conn.cursor(dictionary=True)
                cursor.execute(
                    """
                    SELECT log_id, timestamp, object_class,
                           ROUND(confidence, 2) AS confidence,
                           bbox_x, bbox_y, bbox_w, bbox_h
                    FROM detection_logs
                    ORDER BY log_id DESC
                    LIMIT %s
                    """,
                    (int(limit),),
                )
                rows = cursor.fetchall()
                cursor.close()
                conn.close()
                return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
            except Error:
                self.use_mysql = False

        conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        df = pd.read_sql_query(
            f"""
            SELECT log_id, timestamp, object_class,
                   ROUND(confidence, 2) AS confidence,
                   bbox_x, bbox_y, bbox_w, bbox_h
            FROM detection_logs
            ORDER BY log_id DESC
            LIMIT {int(limit)}
            """,
            conn,
        )
        conn.close()
        return df if not df.empty else pd.DataFrame(columns=cols)

    def get_summary_metrics(self) -> dict:
        df = self.fetch_recent_logs(limit=1000)
        if df.empty:
            return {"total_detections": 0, "top_class": "None", "avg_confidence": 0.0}
        top_cls = df["object_class"].value_counts().idxmax()
        top_cnt = int(df["object_class"].value_counts().max())
        avg_conf = round(float(df["confidence"].mean()) * 100, 1)
        return {
            "total_detections": len(df),
            "top_class": f"{top_cls} ({top_cnt})",
            "avg_confidence": avg_conf,
        }
