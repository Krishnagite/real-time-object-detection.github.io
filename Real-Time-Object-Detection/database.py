import os
import time
import threading

import pandas as pd
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DatabaseManager:
    """
    Handles MySQL database connections, automatic schema initialization,
    thread-safe throttled event logging, and real-time analytics queries.
    """

    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.user = os.getenv("DB_USER", "root")
        self.password = os.getenv("DB_PASSWORD", "")
        self.database = os.getenv("DB_NAME", "vision_platform")
        self.port = int(os.getenv("DB_PORT", 3306))

        self._lock = threading.Lock()
        self._last_logged_time = {}  # Tracks last log timestamp per object_class
        self.is_connected = False

        self.init_db()

    def _get_server_connection(self):
        """Connects to the MySQL server without specifying a database."""
        return mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            port=self.port,
            connection_timeout=5,
        )

    def _get_db_connection(self):
        """Connects directly to the vision_platform database."""
        return mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            port=self.port,
            connection_timeout=5,
        )

    def init_db(self) -> bool:
        """
        Automatically creates the `vision_platform` database and
        `detection_logs` table if they do not already exist.
        """
        try:
            conn = self._get_server_connection()
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
                    bbox_h INT NOT NULL,
                    INDEX idx_timestamp (timestamp DESC),
                    INDEX idx_object_class (object_class)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            conn.commit()
            cursor.close()
            conn.close()
            self.is_connected = True
            return True
        except Error as e:
            print(f"[Database Warning] Could not initialize MySQL database: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"[Database Error] Unexpected error during init_db: {e}")
            self.is_connected = False
            return False

    def check_connection(self) -> bool:
        """Verifies whether the MySQL server and database are reachable."""
        try:
            conn = self._get_db_connection()
            if conn.is_connected():
                conn.close()
                self.is_connected = True
                return True
        except Exception:
            pass
        self.is_connected = False
        return False

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
        """
        Inserts a detection event into `detection_logs` using parameterized SQL.
        Applies per-class cooldown throttling so 30 FPS video does not flood MySQL.
        """
        now = time.time()
        with self._lock:
            last_time = self._last_logged_time.get(object_class, 0.0)
            if (now - last_time) < cooldown_seconds:
                return False
            self._last_logged_time[object_class] = now

        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            query = """
                INSERT INTO detection_logs
                (object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            values = (
                str(object_class),
                round(float(confidence), 4),
                int(bbox_x),
                int(bbox_y),
                int(bbox_w),
                int(bbox_h),
            )
            cursor.execute(query, values)
            conn.commit()
            cursor.close()
            conn.close()
            self.is_connected = True
            return True
        except Error as e:
            print(f"[Database Insert Error] Failed to log detection: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"[Unexpected DB Error] {e}")
            self.is_connected = False
            return False

    def fetch_recent_logs(self, limit: int = 20) -> pd.DataFrame:
        """
        Returns the most recent detection logs as a Pandas DataFrame
        for clean rendering in the Streamlit dashboard.
        """
        columns = [
            "log_id",
            "timestamp",
            "object_class",
            "confidence",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
        ]
        try:
            conn = self._get_db_connection()
            query = """
                SELECT log_id, timestamp, object_class,
                       ROUND(confidence, 2) AS confidence,
                       bbox_x, bbox_y, bbox_w, bbox_h
                FROM detection_logs
                ORDER BY log_id DESC
                LIMIT %s
            """
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (int(limit),))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            self.is_connected = True
            if not rows:
                return pd.DataFrame(columns=columns)
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"[Database Fetch Error] {e}")
            self.is_connected = False
            return pd.DataFrame(columns=columns)

    def get_summary_metrics(self) -> dict:
        """
        Queries total logged detections, most frequent object class,
        and average confidence score for dashboard KPI cards.
        """
        metrics = {
            "total_detections": 0,
            "top_class": "None",
            "avg_confidence": 0.0,
        }
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute(
                """
                SELECT COUNT(*) AS total,
                        COALESCE(AVG(confidence), 0.0) AS avg_conf
                FROM detection_logs
                """
            )
            row = cursor.fetchone()
            if row:
                metrics["total_detections"] = int(row["total"])
                metrics["avg_confidence"] = round(float(row["avg_conf"]) * 100, 1)

            cursor.execute(
                """
                SELECT object_class, COUNT(*) AS cnt
                FROM detection_logs
                GROUP BY object_class
                ORDER BY cnt DESC
                LIMIT 1
                """
            )
            top_row = cursor.fetchone()
            if top_row and top_row["object_class"]:
                metrics["top_class"] = f"{top_row['object_class']} ({top_row['cnt']})"

            cursor.close()
            conn.close()
            self.is_connected = True
        except Exception as e:
            print(f"[Database Metrics Error] {e}")
            self.is_connected = False

        return metrics
