from fastapi import FastAPI

app = FastAPI()

@app.get("/hello")
def say_hello():
    return f"Hello, World!"

@app.post("/hello_name")
def hello_name(name: str):
    return f"Hello, {name}!"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
