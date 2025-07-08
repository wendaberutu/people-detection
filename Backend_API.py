from flask import Flask, request, jsonify
from flask_cors import CORS
import pymysql
from pymysql.cursors import DictCursor

app = Flask(__name__)
CORS(app)

# --- KONFIGURASI DATABASE ---
db_config = {
    "host": "127.0.0.1",
    "port": 3307,
    "user": "root",
    "password": "",  # ganti kalau pakai password
    "database": "cctv_db",
    "cursorclass": DictCursor
}


def get_db_connection():
    return pymysql.connect(**db_config)

# --- CEK KONEKSI ---
@app.route('/')
def index():
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT VERSION()")
            version = cur.fetchone()
        conn.close()
        return jsonify({"message": "Connected!", "version": version['VERSION()']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ENDPOINT LAIN BISA MENYUSUL ---
if __name__ == '__main__':
    app.run(debug=True)
