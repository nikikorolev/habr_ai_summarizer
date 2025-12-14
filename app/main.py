from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import time
import json
import base64
from typing import List, Optional
import sqlite3

app = FastAPI()

class InputData(BaseModel):
    features: List[float]
    text: Optional[str] = None

class SimpleModel:
    def predict(self, data: dict):
        return {"result": [0.5, 0.3, 0.2], "success": True}
    
    def process_image(self, image_bytes):
        return {"size": len(image_bytes), "processed": True}

model = SimpleModel()

def init_db():
    conn = sqlite3.connect('requests.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            success BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()

init_db()


@app.post("/forward")
async def forward(data: InputData):
    try:
        start_time = time.time()
        
        conn = sqlite3.connect('requests.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO history (endpoint, success) VALUES (?, ?)",
            ("/forward", True)
        )
        conn.commit()
        conn.close()
        
        result = model.predict(data.dict())
        
        processing_time = time.time() - start_time
        
        return {
            **result,
            "processing_time": processing_time
        }
        
    except Exception:
        raise HTTPException(status_code=400, detail="bad request")



@app.get("/history")
async def get_history():
    conn = sqlite3.connect('requests.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT 100")
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "endpoint": row[1],
            "timestamp": row[2],
            "success": bool(row[3])
        })
    
    return {"history": history}

@app.delete("/history")
async def delete_history(
    confirm_token: str = Header(None, alias="X-Confirm-Token")
):
    if confirm_token != "mysecret123":
        raise HTTPException(status_code=401, detail="Invalid token")
    
    conn = sqlite3.connect('requests.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history")
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    
    return {"deleted": deleted_count}

