from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import uvicorn
import joblib

app = FastAPI()

knn=joblib.load('model.joblib')

# 경도가 x, 위도가 y
@app.post("/")
def get_result(x: str = Form(...), y: str = Form(...)):
    fx = float(x)
    fy = float(y)
    
    if 33.2533 < fy < 38.37881 and 126.10863 < fx < 129.365:#한국 안에 있다면?(경도 위도 출처 : chatgpt)
        prediction = knn.predict([[fx, fy]])
        local = prediction[0]
    else:
        local = '외국'
    return JSONResponse({'local': local})