{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "edea75db-4caa-474b-9182-16f8b5610b62",
   "metadata": {},
   "outputs": [],
   "source": [
    "import nest_asyncio\n",
    "nest_asyncio.apply()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fe63927a-41e9-4dcb-85f7-b0431ffc7f3f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastapi import FastAPI, File, UploadFile\n",
    "from fastapi.middleware.cors import CORSMiddleware\n",
    "import uvicorn\n",
    "import numpy as np\n",
    "from io import BytesIO\n",
    "from PIL import Image\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aeb4a79d-9f98-4df2-9ec6-507e13b5fc72",
   "metadata": {},
   "outputs": [],
   "source": [
    "app = FastAPI()\n",
    "\n",
    "origins = [\n",
    "    \"http://localhost\",\n",
    "    \"http://localhost:3000\",\n",
    "]\n",
    "app.add_middleware(\n",
    "    CORSMiddleware,\n",
    "    allow_origins=origins,\n",
    "    allow_credentials=True,\n",
    "    allow_methods=[\"*\"],\n",
    "    allow_headers=[\"*\"],\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "32953f6c-2a10-4c2d-ada2-acb1e334fc58",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    }
   ],
   "source": [
    "MODEL = tf.keras.models.load_model(\"../saved_models/2.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4bd10d50-26a2-4967-876e-ee080d7667a7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d9b843d5-ef4e-485a-9395-27eb22f32a29",
   "metadata": {},
   "outputs": [],
   "source": [
    "CLASS_NAMES = [\"Early Blight\", \"Late Blight\", \"Healthy\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a5a6141c-24a1-42aa-8820-6abb0955e2bd",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "750c0d16-4149-48b6-a136-32966d1d66c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.get(\"/ping\")\n",
    "async def ping():\n",
    "    return \"Hello, I am alive\"\n",
    "\n",
    "def read_file_as_image(data) -> np.ndarray:\n",
    "    image = np.array(Image.open(BytesIO(data)))\n",
    "    return image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d34b5783-5a55-4f8a-9b82-ad2cfade3d30",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "27e38a61-286a-48bf-9e4f-27595bb701fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.post(\"/predict\")\n",
    "async def predict(\n",
    "    file: UploadFile = File(...)\n",
    "):\n",
    "    image = read_file_as_image(await file.read())\n",
    "    img_batch = np.expand_dims(image, 0)\n",
    "    \n",
    "    predictions = MODEL.predict(img_batch)\n",
    "\n",
    "    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]\n",
    "    confidence = np.max(predictions[0])\n",
    "    return {\n",
    "        'class': predicted_class,\n",
    "        'confidence': float(confidence)\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "730de5cb-9ca1-4ac3-b691-25b7fb196b8b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:     Started server process [27784]\n",
      "INFO:     Waiting for application startup.\n",
      "INFO:     Application startup complete.\n",
      "INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 673ms/step\n",
      "INFO:     ::1:57965 - \"POST /predict HTTP/1.1\" 200 OK\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 557ms/step\n",
      "INFO:     ::1:50152 - \"POST /predict HTTP/1.1\" 200 OK\n",
      "INFO:     ::1:50577 - \"POST /predict HTTP/1.1\" 500 Internal Server Error\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "ERROR:    Exception in ASGI application\n",
      "Traceback (most recent call last):\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\uvicorn\\protocols\\http\\h11_impl.py\", line 403, in run_asgi\n",
      "    result = await app(  # type: ignore[func-returns-value]\n",
      "             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\uvicorn\\middleware\\proxy_headers.py\", line 60, in __call__\n",
      "    return await self.app(scope, receive, send)\n",
      "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\fastapi\\applications.py\", line 1054, in __call__\n",
      "    await super().__call__(scope, receive, send)\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\starlette\\applications.py\", line 112, in __call__\n",
      "    await self.middleware_stack(scope, receive, send)\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\starlette\\middleware\\errors.py\", line 187, in __call__\n",
      "    raise exc\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\starlette\\middleware\\errors.py\", line 165, in __call__\n",
      "    await self.app(scope, receive, _send)\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\starlette\\middleware\\cors.py\", line 85, in __call__\n",
      "    await self.app(scope, receive, send)\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\starlette\\middleware\\exceptions.py\", line 62, in __call__\n",
      "    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\starlette\\_exception_handler.py\", line 53, in wrapped_app\n",
      "    raise exc\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\starlette\\_exception_handler.py\", line 42, in wrapped_app\n",
      "    await app(scope, receive, sender)\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\starlette\\routing.py\", line 714, in __call__\n",
      "    await self.middleware_stack(scope, receive, send)\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\starlette\\routing.py\", line 734, in app\n",
      "    await route.handle(scope, receive, send)\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\starlette\\routing.py\", line 288, in handle\n",
      "    await self.app(scope, receive, send)\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\starlette\\routing.py\", line 76, in app\n",
      "    await wrap_app_handling_exceptions(app, request)(scope, receive, send)\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\starlette\\_exception_handler.py\", line 53, in wrapped_app\n",
      "    raise exc\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\starlette\\_exception_handler.py\", line 42, in wrapped_app\n",
      "    await app(scope, receive, sender)\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\starlette\\routing.py\", line 73, in app\n",
      "    response = await f(request)\n",
      "               ^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\fastapi\\routing.py\", line 301, in app\n",
      "    raw_response = await run_endpoint_function(\n",
      "                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\fastapi\\routing.py\", line 212, in run_endpoint_function\n",
      "    return await dependant.call(**values)\n",
      "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\likit\\AppData\\Local\\Temp\\ipykernel_27784\\2966514226.py\", line 31, in predict\n",
      "    image = read_file_as_image(await file.read())\n",
      "            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\likit\\AppData\\Local\\Temp\\ipykernel_27784\\1203864892.py\", line 6, in read_file_as_image\n",
      "    image = np.array(Image.open(BytesIO(data)))\n",
      "                     ^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\likit\\anaconda3\\Lib\\site-packages\\PIL\\Image.py\", line 3498, in open\n",
      "    raise UnidentifiedImageError(msg)\n",
      "PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x0000029D4EDE9D50>\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 440ms/step\n",
      "INFO:     ::1:53784 - \"POST /predict HTTP/1.1\" 200 OK\n"
     ]
    }
   ],
   "source": [
    "if __name__ == \"__main__\":\n",
    "    uvicorn.run(app, host='localhost', port=8000)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
